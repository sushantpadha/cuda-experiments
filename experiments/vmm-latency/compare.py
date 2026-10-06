#!/usr/bin/env python3
"""runs ./alloc_compare 5 times -> results/alloc_compare_raw.csv, results/alloc_compare.csv,
results/vs_cudamalloc.png, results/vs_cudamallocmanaged.png
usage: compare.py [--plot]   (--plot: reuse the raw csv, skip the runs)"""
import csv
import statistics as st
import subprocess
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS, REPS = 5, 10
RAW = "results/alloc_compare_raw.csv"
TITLE = "RTX 4050 Laptop, driver 580, CUDA 13.0; host timer, calls spaced 20 ms apart"

# ---- run (or reuse) ----
if "--plot" not in sys.argv:
    subprocess.run(["make", "-s", "alloc_compare"], check=True)
    with open(RAW, "w") as f:
        for r in range(RUNS):
            out = subprocess.run(["./alloc_compare", str(r), str(REPS)], check=True, capture_output=True, text=True).stdout
            f.write(out if r == 0 else out.split("\n", 1)[1])
            print(f"run {r} done", file=sys.stderr)

# ---- each compared quantity = sum of measured phases for one api ----
SERIES = {
    "cudaMalloc":                       ("cudaMalloc", ["alloc"]),
    "cudaFree (cudaMalloc)":            ("cudaMalloc", ["free"]),
    "VMM create + map + set access":    ("vmm", ["alloc", "set_access"]),
    "VMM unmap + release":              ("vmm", ["free"]),
    "cudaMallocManaged + GPU touch":    ("cudaMallocManaged", ["alloc", "touch"]),
    "cudaMallocManaged alone":          ("cudaMallocManaged", ["alloc"]),
    "cudaFree (managed, touched)":      ("cudaMallocManaged", ["free"]),
    "VMM reserve + create + map + set access": ("vmm", ["reserve", "alloc", "set_access"]),
    "VMM unmap + release + address free":      ("vmm", ["free", "addr_free"]),
}
samples = defaultdict(lambda: defaultdict(list))   # samples[(series, size)][run] -> values
for row in csv.DictReader(open(RAW)):
    for name, (api, phases) in SERIES.items():
        if row["api"] == api:
            samples[(name, int(row["size_mib"]))][int(row["run"])].append(sum(float(row[f"{p}_us"]) for p in phases))
sizes = sorted({k[1] for k in samples})

# ---- summary: median over all samples, plus spread of the per-run medians ----
rows = []
for name in SERIES:
    for s in sizes:
        per_run = samples[(name, s)]
        allv = [x for v in per_run.values() for x in v]
        meds = [st.median(v) for v in per_run.values()]
        rows.append(dict(series=name, size_mib=s, n=len(allv), runs=len(meds), median_us=round(st.median(allv), 1),
                         run_median_min_us=round(min(meds), 1), run_median_max_us=round(max(meds), 1)))
with open("results/alloc_compare.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
med = {(r["series"], r["size_mib"]): r["median_us"] for r in rows}

# ---- plots: one figure per comparison, allocate and free side by side ----
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "legend.fontsize": 9})


def line(ax, name, color, ls="-", dots=True):
    if dots:   # faint dots: each run's median
        for s in sizes:
            ax.scatter([s] * RUNS, [st.median(v) for v in samples[(name, s)].values()],
                       s=14, color=color, alpha=0.4, linewidths=0)
    ax.plot(sizes, [med[(name, s)] for s in sizes], color=color, marker="o", ms=6, lw=2, ls=ls,
            markeredgecolor="white", markeredgewidth=0.8, label=name)


FIGS = [
    ("vs_cudamalloc", "Steady state: cudaMalloc vs. VMM with the address range already reserved",
     [("allocate", [("cudaMalloc", ORANGE, "-", True), ("VMM create + map + set access", BLUE, "-", True)]),
      ("free", [("cudaFree (cudaMalloc)", ORANGE, "-", True), ("VMM unmap + release", BLUE, "-", True)])]),
    ("vs_cudamallocmanaged", "From scratch: cudaMallocManaged + one GPU touch vs. the full VMM sequence",
     [("allocate", [("cudaMallocManaged + GPU touch", ORANGE, "-", True),
                    ("cudaMallocManaged alone", ORANGE, ":", False),
                    ("VMM reserve + create + map + set access", BLUE, "-", True)]),
      ("free", [("cudaFree (managed, touched)", ORANGE, "-", True),
                ("VMM unmap + release + address free", BLUE, "-", True)])]),
]
for out, title, panels in FIGS:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, constrained_layout=True)
    for ax, (ptitle, series) in zip(axes, panels):
        for args in series:
            line(ax, *args)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("allocation size (MiB)")
        ax.set_title(ptitle, loc="left", color=INK)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("latency (µs)\nline = median, dots = median of each run")
    fig.suptitle(f"{title} (device memory; {RUNS} runs x {REPS} reps)\n{TITLE}", x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(f"results/{out}.png", dpi=160)

# ---- console summary ----
print(f"median µs over {RUNS} runs x {REPS} reps")
for name in SERIES:
    print(f"{name:42}" + "".join(f"{med[(name, s)]:>10.1f}" for s in sizes))
print(f"{'MiB':42}" + "".join(f"{s:>10}" for s in sizes))
print("wrote results/alloc_compare_raw.csv, results/alloc_compare.csv, results/vs_cudamalloc.png, results/vs_cudamallocmanaged.png")
