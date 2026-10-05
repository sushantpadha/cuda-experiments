#!/usr/bin/env bash
# build, trace with nsys, export, analyse. usage: ./run.sh [reps]
#   spaced:       20 ms busy-wait between reps (driver background work finishes first)
#   backtoback:   no gap (background work from earlier frees spills into later calls)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
make -s
: > results/nsys.log
for mode in spaced backtoback; do
    gap=$([ "$mode" = spaced ] && echo 20 || echo 0)
    nsys profile -t cuda,nvtx -o "results/$mode" --force-overwrite true ./vmm_latency "${1:-20}" "$gap" >> results/nsys.log 2>&1
    nsys export --type sqlite --force-overwrite true -o "results/$mode.sqlite" "results/$mode.nsys-rep" >> results/nsys.log 2>&1
done
python3 analyze.py results/spaced.sqlite results/backtoback.sqlite
