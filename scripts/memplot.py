#!/usr/bin/env python3
"""memplot: draw one or more memtrace CSVs on one plot.

    scripts/memplot.py -o analysis/NAME.png [--title T] [--right COLUMN] [--right-for LABEL,...] LABEL=CSV [LABEL=CSV ...]

Each LABEL=CSV is one run and gets its own color. The left axis shows VRAM (solid) and DRAM (dashed) in MiB above
their values at the run's start, with a faint dotted line at each peak. --right COLUMN draws that count column
(for example "hints accepted") on the right axis as a dash-dot line, for the runs named in --right-for (default:
every run that has the column). Time 0 is each command's start. Needs matplotlib.
"""
import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"]


def load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
    num = lambda v: float(v) if v not in ("", "None") else math.nan
    return {k: [num(row[k]) for row in rows] for k in r.fieldnames}


def delta(col):
    """values minus the first real sample, so runs with different idle use line up"""
    base = next((v for v in col if v == v), 0.0)
    return [v - base for v in col]


def main(a):
    runs = [(spec.split("=", 1)[0], load(spec.split("=", 1)[1])) for spec in a.runs]
    right_for = set(a.right_for.split(",")) if a.right_for else None
    fig, ax = plt.subplots(figsize=(11, 6))
    ax2 = ax.twinx() if a.right else None
    tmax = max(max(d["t_s"]) for _, d in runs)
    handles = []
    top = 1   # right axis: at least 0..1, so an all-zero count still gets whole-number ticks
    for i, (label, d) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        t = d["t_s"]
        for col, style in (("vram_mib", "-"), ("dram_mib", "--")):
            y = delta(d[col])
            ax.plot(t, y, style, color=c, linewidth=1.6)
            peak = max(v for v in y if v == v)
            ax.axhline(peak, color=c, linestyle=":", linewidth=1, alpha=0.4)
            ax.text(tmax, peak, f" {peak:.0f}", color=c, alpha=0.7, fontsize="x-small", va="bottom", ha="left")
        handles.append(Line2D([], [], color=c, linewidth=3, label=label))
        if ax2 and a.right in d and (right_for is None or label in right_for):
            ax2.step(t, d[a.right], where="post", linestyle="-.", color=c, linewidth=1.4)
            top = max(top, max(v for v in d[a.right] if v == v))
    handles += [Line2D([], [], color="black", linestyle="-", label="VRAM"),
                Line2D([], [], color="black", linestyle="--", label="DRAM"),
                Line2D([], [], color="black", linestyle=":", alpha=0.5, label="peak")]
    if ax2:
        handles.append(Line2D([], [], color="black", linestyle="-.", label=f"{a.right} (right axis)"))
        ax2.set_ylabel(a.right)
        ax2.set_ylim(0, top * 1.1)
        ax2.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("time since start (s)")
    ax.set_ylabel("memory used above the start (MiB)")
    ax.set_xlim(0, tmax * 1.06)
    ax.set_title(a.title)
    fig.legend(handles=handles, loc="lower center", fontsize="small", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))   # room for the legend under the axes
    fig.savefig(a.output, dpi=130)
    print(f"memplot: {a.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="overlay memtrace CSVs: VRAM and DRAM on the left axis, a count on the right")
    ap.add_argument("-o", "--output", required=True, help="output PNG")
    ap.add_argument("--title", default="")
    ap.add_argument("--right", help="count column to draw on the right axis")
    ap.add_argument("--right-for", help="comma-separated labels that get the right-axis line (default: all)")
    ap.add_argument("runs", nargs="+", metavar="LABEL=CSV")
    main(ap.parse_args())
