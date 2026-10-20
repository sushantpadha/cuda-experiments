#!/usr/bin/env python3
"""Multi-process stress test: make test-workers

Starts one vmemd, then several bin/worker processes at once (no MPS). Every worker runs the stress.cu workload on its
own matrix size and its own images. The script checks that:
  - every worker finishes with exit code 0 and no wrong spot checks;
  - each worker's checksum equals the checksum it gets when it runs alone (the "solo" pass);
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
RESULT = re.compile(r"^RESULT (\S+) (-?\d+) (\S+) ([\d.]+) ([\d.]+) (\d+) (\d+)$", re.M)

# worker i: its own matrix size and image size; it also starts at its own base image
MATRIX_N = [2048, 3072, 4096, 5120]
IMAGE_MIB = [128, 192, 256, 320]


def spec(i):
    return dict(name=f"worker{i}", n=MATRIX_N[i % len(MATRIX_N)], image_mib=IMAGE_MIB[i % len(IMAGE_MIB)], first_base=i)


class Worker:
    def __init__(self, tmp, broker, resident, images, **p):
        self.name = p["name"]
        self.out_path = os.path.join(tmp, self.name + ".out")
        self.out = open(self.out_path, "w")
        cmd = [WORKER, "--name", self.name, "--socket", broker.sock, "--n", str(p["n"]), "--image-mib", str(p["image_mib"]),
               "--images", str(images), "--first-base", str(p["first_base"]), "--data", DATA]
        if resident:
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
    workers = [Worker(tmp, broker, args.resident, args.images, **s) for s in specs]
    try:
        if wait_all(workers, Worker.ready, 120, "the workers to connect"):
            for w in workers:
                w.go()
            wait_all(workers, lambda w: w.proc.poll() is not None, args.timeout, "the workers to finish")
    finally:
        for w in workers:
            h.kill_group(w.proc)   # no-op for those that finished
            w.out.close()
    return {w.name: (w.proc.returncode, w.text()) for w in workers}


def parse(output):
    m = RESULT.search(output)
    if not m:
        return None
    name, rc, checksum, t_mm, t_img, ok, refused = m.groups()
    return dict(rc=int(rc), checksum=checksum, t_mm=float(t_mm), t_img=float(t_img), hints_ok=int(ok), hints_no=int(refused))


def broker_peaks(lines):
    """most MiB the broker ever counted over all tenants, device and host"""
    dev = host = 0
    for line in lines:
        if m := h.STATE.search(line):
            dev, host = max(dev, int(m.group(4))), max(host, int(m.group(5)))
    return dev, host


def main(args):
    specs = [spec(i) for i in range(args.workers)]

    def body(tmp):
        broker = h.Broker(tmp, f"vram_limit_mib = {args.vram_limit}\ndram_limit_mib = {args.dram_limit}\n"
                               f"ctx_overhead_mib = {args.ctx_overhead}\n").start()
        print(f"{args.workers} workers, {args.images} images each{', resident' if args.resident else ''}; "
              f"broker limits {args.vram_limit} MiB device, {args.dram_limit} MiB host")

        solo = {}
        if args.solo:
            h.phase("solo", "each worker alone, for the reference checksums")
            for s in specs:
                (name, (rc, out)), = run_workers(tmp, broker, args, [s]).items()
                r = parse(out)
                h.expect(rc == 0 and r is not None, f"{name} alone: exit code 0")
                if r:
                    solo[name] = r

        h.phase("together", "all workers at once")
        start = len(broker.lines())
        t0 = time.time()
        results = run_workers(tmp, broker, args, specs)
        wall = time.time() - t0

        rows = []
        for name, (rc, out) in results.items():
            r = parse(out)
            h.expect(rc == 0 and r is not None and r["rc"] == 0, f"{name}: exit code 0 (got {rc})")
            bad = re.findall(r"(\d+) of \d+ spot checks wrong", out)
            h.expect(not any(int(b) for b in bad), f"{name}: no wrong spot checks")
            if r and name in solo:
                h.expect(r["checksum"] == solo[name]["checksum"], f"{name}: checksum equals the solo checksum ({r['checksum']})")
            if r:
                rows.append((name, r))
            else:
                print(out[-800:])

        # the broker handles the last disconnects after the workers are gone ("connect: pid" also matches "disconnect: pid")
        lines = h.wait_log(broker.log, lambda ls: 2 * h.count(ls, "disconnect:") == h.count(ls, "connect: pid"), 10) \
            or broker.lines()
        h.expect(broker.alive(), "vmemd is still running")
        for s in specs:
            h.expect(any(f'is "{s["name"]}"' in line for line in lines), f"vmemd saw {s['name']}")
        h.check_sums(lines)
        h.check_refusals(lines)
        h.check_totals_zero(lines)

        dev, host = broker_peaks(lines)
        print(f"\n{'worker':<9}{'matmul s':>10}{'sharpen s':>11}{'hints ok':>10}{'refused':>9}")
        for name, r in rows:
            print(f"{name:<9}{r['t_mm']:>10.2f}{r['t_img']:>11.2f}{r['hints_ok']:>10}{r['hints_no']:>9}")
        h.note(f"all together took {wall:.1f} s; most the broker counted at once: {dev} MiB device, {host} MiB host")
        h.events(lines, start)

    return h.run(body, "vmemd-workers-", args.timeout * (args.workers + 1) + 300)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="multi-process stress test against one vmemd")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--images", type=int, default=6, help="images per worker")
    ap.add_argument("--no-resident", dest="resident", action="store_false",
                    help="one image at a time instead of all at once with evict/prefetch hints")
    ap.add_argument("--no-solo", dest="solo", action="store_false", help="skip the one-at-a-time pass (no checksum reference)")
    ap.add_argument("--vram-limit", type=int, default=5400, help="MiB the broker hands out on the device")
    ap.add_argument("--dram-limit", type=int, default=3000, help="MiB the broker hands out in pinned host memory")
    ap.add_argument("--ctx-overhead", type=int, default=100, help="MiB of VRAM each worker's context takes")
    ap.add_argument("--timeout", type=int, default=600, help="seconds the workers may run")
    a = ap.parse_args()
    if not h.need(h.VMEMD, WORKER):
        sys.exit(2)
    sys.exit(main(a))
