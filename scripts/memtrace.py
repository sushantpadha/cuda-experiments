#!/usr/bin/env python3
"""memtrace: run any command and record VRAM and DRAM use over time, plus counts of log lines, as a CSV.

    scripts/memtrace.py [-o analysis/NAME.csv] [--interval S] [--gpu N] [--count LABEL=GLOB:REGEX ...] -- CMD [ARGS ...]

It writes one CSV; scripts/memplot.py draws it. Every sample has the time since the start, the GPU's used memory (device-wide,
from nvidia-smi), the machine's used DRAM (MemTotal - MemAvailable, so pinned memory counts too) and one column per
--count. A count is the number of lines matching REGEX so far in the files matching GLOB that were written after the
command started; the glob is re-read every sample, so it can name a file the command creates later.

Example (vmemd refusals from a broker log):
    scripts/memtrace.py -o analysis/run1.csv --count "hints refused=/tmp/vmemd-*/vmemd.log:move .* refused" -- ./my_app
Needs nvidia-smi only. Exits with the command's exit code.
"""
import argparse
import csv
import glob
import os
import re
import signal
import subprocess
import sys
import threading
import time


# ---- sources ----


class Vram:
    """one nvidia-smi in loop mode; keeps the latest used MiB"""

    def __init__(self, gpu, interval):
        ms = max(int(interval * 1000), 10)
        self.p = subprocess.Popen(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                   f"-lms={ms}"], stdout=subprocess.PIPE, text=True)
        self.mib = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            try:
                self.mib = int(line.strip())
            except ValueError:
                pass

    def stop(self):
        self.p.terminate()


def dram_mib():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) // 1024


class Count:
    """lines matching regex in files matching pattern, written after t0; reads each file incrementally"""

    def __init__(self, spec, t0):
        self.label, rest = spec.split("=", 1)
        self.pattern, rx = rest.split(":", 1)
        self.rx, self.t0 = re.compile(rx), t0
        self.seen = {}   # path -> (offset, count)

    def value(self):
        for path in glob.glob(self.pattern):
            try:
                if path not in self.seen and os.path.getmtime(path) < self.t0:
                    continue   # left over from an earlier run
                off, n = self.seen.get(path, (0, 0))
                with open(path, "rb") as f:
                    f.seek(off)
                    data = f.read()
                cut = data.rfind(b"\n") + 1   # only whole lines; the rest is read next time
                n += sum(1 for line in data[:cut].decode(errors="replace").splitlines() if self.rx.search(line))
                self.seen[path] = (off + cut, n)
            except OSError:
                pass
        return sum(n for _, n in self.seen.values())


# ---- record ----


def record(args):
    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t0 = time.time()
    vram = Vram(args.gpu, args.interval)
    counts = [Count(c, t0) for c in args.count]
    while vram.mib is None and time.time() - t0 < 5:   # first nvidia-smi line
        time.sleep(0.01)
    t0 = time.time()   # time 0 is the command's start; counts still take files written since nvidia-smi started
    rows = []
    cmd = subprocess.Popen(args.cmd)
    signal.signal(signal.SIGINT, lambda *_: cmd.send_signal(signal.SIGINT))   # Ctrl-C goes to the command
    try:
        while True:
            done = cmd.poll() is not None
            rows.append([round(time.time() - t0, 3), vram.mib, dram_mib()] + [c.value() for c in counts])
            if done:
                break
            time.sleep(args.interval)
    finally:
        vram.stop()
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")   # csv defaults to CRLF
        w.writerow(["t_s", "vram_mib", "dram_mib"] + [c.label for c in counts])
        w.writerows(rows)
    print(f"memtrace: {len(rows)} samples in {out}")
    return cmd.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="record VRAM, DRAM and log-line counts while a command runs, as a CSV")
    ap.add_argument("-o", "--output", default="analysis/memtrace.csv", help="output CSV (default analysis/memtrace.csv)")
    ap.add_argument("--interval", type=float, default=0.1, help="seconds between samples (default 0.1)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--count", action="append", default=[], metavar="LABEL=GLOB:REGEX", help="count matching log lines; repeatable")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command to run")
    a = ap.parse_args()
    a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not a.cmd:
        ap.error("give a command after --")
    sys.exit(record(a))
