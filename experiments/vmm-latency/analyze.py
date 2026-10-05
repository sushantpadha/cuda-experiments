#!/usr/bin/env python3
"""nsys sqlite exports -> results/latency.csv, results/vmm_latency.png, results/vmm_backtoback.png
usage: analyze.py <spaced.sqlite> <backtoback.sqlite>"""
import csv
import sqlite3
import statistics as st
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPS = ["cuMemAddressReserve", "cuMemCreate", "cuMemMap", "cuMemSetAccess",
       "cuMemUnmap", "cuMemRelease", "cuMemAddressFree"]
TITLE = "RTX 4050 Laptop, driver 580, CUDA 13.0; nsys driver-API trace"


# ---- load: each driver call, labelled by the nvtx range it falls in ----
def load(path):
    db = sqlite3.connect(path)
    ranges = db.execute("select start, end, text from NVTX_EVENTS where text like '%|%|%|%' order by start").fetchall()
    calls = db.execute("""select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r
                          join StringIds s on r.nameId = s.id where s.value like 'cuMem%' order by r.start""").fetchall()
    out, i = defaultdict(list), 0
    for c0, c1, op in calls:
        while i < len(ranges) and ranges[i][1] < c0:
            i += 1
        if i == len(ranges) or not (ranges[i][0] <= c0 and c1 <= ranges[i][1]):
            continue
        loc, mib, perm, rep = ranges[i][2].split("|")
        if int(rep) >= 0:   # rep -1 = warmup
            out[(loc, int(mib), perm, op)].append((c1 - c0) / 1e3)
    return out


data = {"spaced": load(sys.argv[1]), "backtoback": load(sys.argv[2])}

# ---- summarise ----
rows = []
for mode, samples in data.items():
    for (loc, mib, perm, op), v in sorted(samples.items()):
        q = st.quantiles(v, n=4)
        rows.append(dict(mode=mode, loc=loc, size_mib=mib, perm=perm, op=op, n=len(v),
                         median_us=round(st.median(v), 2), p25_us=round(q[0], 2), p75_us=round(q[2], 2),
                         max_us=round(max(v), 2)))
with open("results/latency.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
get = {(r["mode"], r["loc"], r["size_mib"], r["perm"], r["op"]): r for r in rows}
sizes = sorted({r["size_mib"] for r in rows})
reps = rows[0]["n"]

# ---- fit latency = a + b * size (spaced, read/write), weighted by 1/median so small sizes count ----
fits = []
for loc in ["device", "host"]:
    for op in OPS:
        x = np.array(sizes, float)
        y = np.array([get[("spaced", loc, s, "rw", op)]["median_us"] for s in sizes])
        A = np.vstack([np.ones_like(x), x]).T / y[:, None]
        (a, b), *_ = np.linalg.lstsq(A, np.ones_like(y), rcond=None)
        pred = a + b * x
        fits.append(dict(loc=loc, op=op, fixed_us=round(a, 1), per_mib_us=round(b, 3),
                         max_rel_err=round(float(np.max(np.abs(pred - y) / y)), 2)))
with open("results/fit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fits[0]))
    w.writeheader()
    w.writerows(fits)

# ---- plot style ----
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SERIES = {"cuMemCreate": ("#2a78d6", "o"), "cuMemMap": ("#eb6834", "s"), "cuMemSetAccess": ("#1baf7a", "^"),
          "cuMemUnmap": ("#eda100", "D"), "cuMemRelease": ("#e87ba4", "v")}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "legend.fontsize": 9})


def style(ax, title):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("allocation size (MiB)")
    ax.set_title(title, loc="left", color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def slope1(ax, y_at_2mib):
    # faint guide: linear growth (10x size -> 10x time)
    x = np.array([sizes[0], sizes[-1]], float)
    ax.plot(x, y_at_2mib * x / sizes[0], color="#b8b7b1", lw=1, ls=":", zorder=0)
    ax.text(256, y_at_2mib * 256 / sizes[0] * 1.6, "slope 1 (linear)", color=MUTED, fontsize=8,
            rotation=24, ha="center", va="bottom")


def line(ax, mode, loc, perm, op, color, marker, ls="-", label=None, x=None):
    pts = [get[(mode, loc, s, perm, op)] for s in sizes]
    y = [p["median_us"] for p in pts]
    err = [[p["median_us"] - p["p25_us"] for p in pts], [p["p75_us"] - p["median_us"] for p in pts]]
    ax.errorbar(x or sizes, y, yerr=err, color=color, marker=marker, ms=6, lw=2, ls=ls, capsize=2, elinewidth=1,
                markeredgecolor="white", markeredgewidth=0.8, label=label or op)


# ---- figure 1: cost of each call, spaced out ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)
for ax, loc, name in [(axes[0], "device", "Device (VRAM)"), (axes[1], "host", "Pinned host (HOST_NUMA)")]:
    for op, (c, m) in SERIES.items():
        line(ax, "spaced", loc, "rw", op, c, m)
    style(ax, f"{name} backing, read/write")
    slope1(ax, 3.0)
axes[0].set_ylabel("latency per call (µs)\nmedian, bars = interquartile range")
axes[0].legend(loc="upper left")
for loc, c in [("device", BLUE), ("host", ORANGE)]:
    for perm, ls, m in [("rw", "-", "o"), ("ro", "--", "s")]:
        line(axes[2], "spaced", loc, perm, "cuMemSetAccess", c, m, ls,
             f"{loc}, {'read/write' if perm == 'rw' else 'read-only'}")
style(axes[2], "cuMemSetAccess: read/write vs read-only")
slope1(axes[2], 30.0)
axes[2].legend(loc="upper left")
fig.suptitle(f"CUDA VMM call latency, calls spaced 20 ms apart ({reps} reps per point)\n{TITLE}",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.savefig("results/vmm_latency.png", dpi=160)

# ---- figure 2: device calls back to back vs spaced ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, constrained_layout=True)
for ax, op in zip(axes, ["cuMemSetAccess", "cuMemUnmap"]):
    for mode, c, dx in [("spaced", BLUE, 0.94), ("backtoback", ORANGE, 1.06)]:
        for s in sizes:
            v = data[mode][("device", s, "rw", op)]
            ax.scatter([s * dx] * len(v), v, s=14, color=c, alpha=0.35, linewidths=0)
        line(ax, mode, "device", "rw", op, c, "o", ls="none", x=[s * dx for s in sizes],
             label="spaced 20 ms apart" if mode == "spaced" else "back to back")
    style(ax, f"{op}, device, read/write")
axes[0].set_ylabel("latency per call (µs)\ndots = every sample, marker = median")
axes[0].legend(loc="upper left")
fig.suptitle(f"Back-to-back calls stall ~2 ms on device mappings\n{TITLE}", x=0.01, ha="left",
             color=INK, fontsize=12)
fig.savefig("results/vmm_backtoback.png", dpi=160)

# ---- console summary ----
print(f"spaced 20 ms apart, read/write, median µs ({reps} reps)")
print(f"{'loc':7}{'MiB':>6}" + "".join(f"{o.replace('cuMem', ''):>15}" for o in OPS))
for loc in ["device", "host"]:
    for s in sizes:
        print(f"{loc:7}{s:>6}" + "".join(f"{get[('spaced', loc, s, 'rw', o)]['median_us']:>15.1f}" for o in OPS))
for op in ["cuMemSetAccess", "cuMemUnmap"]:
    slow = {m: sum(x > 1000 for s in sizes for x in data[m][("device", s, "rw", op)]) for m in data}
    print(f"device {op} calls > 1 ms: spaced {slow['spaced']}, back to back {slow['backtoback']} (of {reps * len(sizes)})")
print("\nfit latency = fixed + per_MiB * size (spaced, read/write)")
for r in fits:
    print(f"{r['loc']:7}{r['op']:22}{r['fixed_us']:>10.1f} us {r['per_mib_us']:>10.3f} us/MiB   max rel err {r['max_rel_err']:.2f}")
print("wrote results/latency.csv, results/fit.csv, results/vmm_latency.png, results/vmm_backtoback.png")
