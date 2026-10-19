#!/usr/bin/env bash
# run test 4 K times, one process per trial, and tally the outcomes
# tests/test4.sh [K] [test4 args...]   e.g. tests/test4.sh 20 64 2000 4
cd "$(dirname "$0")/.." && make -s bin/test4 || exit 1
k=${1:-20}; shift
declare -A n
for ((t = 1; t <= k; t++)); do
    line=$(timeout 60 bin/test4 "$@" 2>&1 | grep RESULT) || line="RESULT: TIMEOUT"
    echo "$t: $line"
    r=${line#RESULT: }; n[${r%% *}]=$((${n[${r%% *}]} + 1))
done
echo "$k trials: CRASH ${n[CRASH]:-0}, CORRUPT ${n[CORRUPT]:-0}, OK ${n[OK]:-0}, other $((k - ${n[CRASH]:-0} - ${n[CORRUPT]:-0} - ${n[OK]:-0}))"
