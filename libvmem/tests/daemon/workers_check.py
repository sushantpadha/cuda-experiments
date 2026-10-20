#!/usr/bin/env python3
"""Multi-process tests: make stress (--workload churn), make test-workers-gemm|image|mix

Starts one vmemd, then several bin/worker processes at once (no MPS). Each worker runs one workload from
tests/workload.cuh: churn (random calls checked after each one, the stress test), gemm (one matrix multiply, its own
matrix size) or image (sharpen on its own images, its own image size); mix alternates gemm and image workers.
With --backend vmem-local, cuda, uvm or raw there is no vmemd and no broker checks: the same run is a baseline for vmem.
The script checks that:
  - every worker finishes with exit code 0 and no wrong spot checks;
  - gemm and image: each worker's checksum equals the checksum it gets when it runs alone (the "solo" pass);
  - churn: every count a worker reported ("view D H") is a state the broker reached, in order;
  - vmemd is still alive afterwards, saw every worker, its totals equal the sum over its tenants after every message,
    and its totals are back to zero.
It prints a table of times and of the most memory the broker ever counted. Workers and broker run in their own
process groups, so a timeout, an error or Ctrl-C kills all of them.
"""
import argparse
import os
import re
import sys
import time

import harness as h

WORKER = h.binary("worker")
DATA = os.path.join(h.ROOT, "tests", "data")
RESULT = re.compile(r"^RESULT (\S+) (-?\d+) (\S+) ([\d.]+) (\d+) (\d+)$", re.M)

# worker i: its own matrix size (gemm) or image size (image); an image worker also starts at its own base image,
# and a churn worker uses seed i. 6 images of these sizes per worker, 6144 MiB in all, never fit in VRAM
MATRIX_N = "8192,10240,12288,14336"
IMAGE_MIB = "160,224,288,352"


def spec(i, args):
    pick = lambda csv: int(csv.split(",")[i % len(csv.split(","))])
    workload = ["gemm", "image"][i % 2] if args.workload == "mix" else args.workload
    return dict(name=f"worker{i}", workload=workload, n=pick(args.matrix_n), image_mib=pick(args.image_mib), first_base=i)


class Worker:
    def __init__(self, tmp, broker, args, **p):
        self.name = p["name"]
        self.out_path = os.path.join(tmp, self.name + ".out")
        self.out = open(self.out_path, "w")
        cmd = [WORKER, "--workload", p["workload"], "--name", self.name, "--backend", args.backend, "--n", str(p["n"]),
               "--image-mib", str(p["image_mib"]), "--images", str(args.images), "--passes", str(args.passes),
               "--first-base", str(p["first_base"]), "--seed", str(p["first_base"]), "--ops", str(args.ops), "--data", DATA]
        if broker:
            cmd += ["--socket", broker.sock]
        if args.resident:
            cmd.append("--resident")
        self.proc = h.popen(cmd, stdin=h.subprocess.PIPE, stdout=self.out, stderr=h.subprocess.STDOUT, text=True)

    def text(self):
        with open(self.out_path) as f:
            return f.read()

    def ready(self):
        return "READY" in self.text() or self.proc.poll() is not None

    def go(self):
        try:
            self.proc.stdin.write("go\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            pass   # the worker died before: its exit code reports it


def wait_all(workers, done, timeout, what):
    """poll until done(w) holds for every worker; false on timeout"""
    end = time.time() + timeout
    while time.time() < end:
        if all(done(w) for w in workers):
            return True
        time.sleep(0.1)
    print(f"  timeout after {timeout} s waiting for {what}")
    return False


def run_workers(tmp, broker, args, specs):
    """start workers, release them together, wait for all; returns {name: (exit code, output)}"""
    workers = [Worker(tmp, broker, args, **s) for s in specs]
    try:
        if wait_all(workers, Worker.ready, 120, "the workers to connect"):
            for w in workers:
                w.go()
            wait_all(workers, lambda w: w.proc.poll() is not None, args.timeout, "the workers to finish")
    finally:
        for w in workers:
            h.kill_group(w.proc)   # no-op for those that finished
            w.out.close()
    return {w.name: (w.proc.returncode, w.text(), w.proc.pid) for w in workers}


def parse(output):
    m = RESULT.search(output)
    if not m:
        return None
    name, rc, checksum, t, ok, refused = m.groups()
    return dict(rc=int(rc), checksum=checksum, t=float(t), hints_ok=int(ok), hints_no=int(refused))


class Views:
    """a finished worker's "view D H" lines, in the shape h.check_views wants"""

    def __init__(self, name, pid, out):
        self.label, self.pid = name, pid
        self.views = [tuple(int(x) for x in l.split()[1:]) for l in out.splitlines() if l.startswith("view ")]


def broker_peaks(lines):
    """most MiB the broker ever counted over all tenants, device and host"""
    dev = host = 0
    for line in lines:
        if m := h.STATE.search(line):
            dev, host = max(dev, int(m.group(4))), max(host, int(m.group(5)))
    return dev, host


def main(args):
    specs = [spec(i, args) for i in range(args.workers)]

    def body(tmp):
        broker = None if args.backend != "vmem" else h.Broker(tmp, f"vram_limit_mib = {args.vram_limit}\ndram_limit_mib = {args.dram_limit}\n"
                               f"ctx_overhead_mib = {args.ctx_overhead}\n").start()
        what = {"churn": f"{args.ops} random calls each", "gemm": f"matrices {args.matrix_n}"}.get(
            args.workload, f"{args.images} images each, {args.passes} passes{', resident' if args.resident else ''}")
        print(f"{args.workers} {args.workload} workers on the {args.backend} backend, {what}"
              + (f"; broker limits {args.vram_limit} MiB device, {args.dram_limit} MiB host" if broker else ""))

        solo = {}
        if args.solo:
            h.phase("solo", "each worker alone, for the reference checksums")
            for s in specs:
                (name, (rc, out, _)), = run_workers(tmp, broker, args, [s]).items()
                r = parse(out)
                h.expect(rc == 0 and r is not None, f"{name} alone: exit code 0")
                if r:
                    solo[name] = r

        h.phase("together", "all workers at once")
        start = len(broker.lines()) if broker else 0
        t0 = time.time()
        results = run_workers(tmp, broker, args, specs)
        wall = time.time() - t0

        rows, views = [], []
        for name, (rc, out, pid) in results.items():
            views.append(Views(name, pid, out))
            r = parse(out)
            h.expect(rc == 0 and r is not None and r["rc"] == 0, f"{name}: exit code 0 (got {rc})")
            if m := re.search(r"churn: .*", out):
                h.note(f"{name}: {m.group(0)}")
            bad = re.findall(r"(\d+) of \d+ spot checks wrong", out)
            h.expect(not any(int(b) for b in bad), f"{name}: no wrong spot checks")
            if r and name in solo:
                h.expect(r["checksum"] == solo[name]["checksum"], f"{name}: checksum equals the solo checksum ({r['checksum']})")
            if r:
                rows.append((name, r))
            else:
                print(out[-800:])

        lines = []
        if broker:
            # the broker handles the last disconnects after the workers are gone ("connect: pid" also matches "disconnect: pid")
            lines = h.wait_log(broker.log, lambda ls: 2 * h.count(ls, "disconnect:") == h.count(ls, "connect: pid"), 10) \
                or broker.lines()
            h.expect(broker.alive(), "vmemd is still running")
            for s in specs:
                h.expect(any(f'is "{s["name"]}"' in line for line in lines), f"vmemd saw {s['name']}")
            h.check_sums(lines)
            h.check_refusals(lines)
            h.check_totals_zero(lines)
            for v in views:
                if v.views:
                    h.check_views(lines, v, args.ctx_overhead)

        dev, host = broker_peaks(lines)
        print(f"\n{'worker':<9}{'time s':>8}{'hints ok':>10}{'refused':>9}")
        for name, r in rows:
            print(f"{name:<9}{r['t']:>8.2f}{r['hints_ok']:>10}{r['hints_no']:>9}")
        h.note(f"all together took {wall:.1f} s" + (f"; most the broker counted at once: {dev} MiB device, {host} MiB host" if broker else ""))
        if broker:
            h.events(lines, start)

    return h.run(body, f"vmemd-{args.workload}-", args.timeout * (args.workers + 1) + 300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="multi-process stress test against one vmemd")
    ap.add_argument("--workload", choices=["churn", "gemm", "image", "mix"], required=True, help="mix: gemm and image workers in turn")
    ap.add_argument("--backend", choices=["vmem", "vmem-local", "cuda", "uvm", "raw"], default="vmem",
                    help="vmem runs with vmemd; vmem-local, cuda, uvm and raw (plain managed memory, no hints) run without it")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--matrix-n", default=MATRIX_N, help="gemm: matrix sizes, one per worker in turn (multiples of 32)")
    ap.add_argument("--image-mib", default=IMAGE_MIB, help="image: image sizes in MiB, one per worker in turn")
    ap.add_argument("--images", type=int, default=6, help="image: images per worker")
    ap.add_argument("--passes", type=int, default=2, help="image: times each worker goes over its images")
    ap.add_argument("--ops", type=int, default=2000, help="churn: random calls per worker")
    ap.add_argument("--no-resident", dest="resident", action="store_false",
                    help="image: one image at a time instead of all at once with evict/prefetch hints")
    ap.add_argument("--no-solo", dest="solo", action="store_false", help="skip the one-at-a-time pass (no checksum reference)")
    ap.add_argument("--vram-limit", type=int, help="MiB the broker hands out on the device (default 5400; churn 1024)")
    ap.add_argument("--dram-limit", type=int, help="MiB the broker hands out in pinned host memory (default 3000; churn 512)")
    ap.add_argument("--ctx-overhead", type=int, default=100, help="MiB of VRAM each worker's context takes")
    ap.add_argument("--timeout", type=int, default=600, help="seconds the workers may run")
    a = ap.parse_args()
    churn = a.workload == "churn"
    a.vram_limit = a.vram_limit or (1024 if churn else 5400)
    a.dram_limit = a.dram_limit or (512 if churn else 3000)
    a.solo = a.solo and not churn   # churn results depend on the other tenants, so there is no reference to compare
    if not h.need(*([h.VMEMD] if a.backend == "vmem" else []), WORKER):
        sys.exit(2)
    sys.exit(main(a))
