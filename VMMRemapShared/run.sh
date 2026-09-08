#!/bin/bash
# Build + run both demos, tee a clean transcript to example_output.txt.
# Usage: ./run.sh
set -e -o pipefail
cd "$(dirname "$0")"
make -s

OUT=example_output.txt

run_demos() {
    echo "### $(date -u +%Y-%m-%dT%H:%M:%SZ)  $(nvcc --version | tail -1)"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | sed 's/^/### GPU: /'
    echo

    echo "======== demo 1: cudaremap primitive (single process) ========"
    ./remap_test 1048576 2

    echo
    echo "======== demo 2: multi-tenant allocator using cudaremap ========"
    rm -f /dev/shm/vmm_share_state /tmp/vmm_share.sock
    ./owner 2 4 2 &
    local owner=$!
    sleep 0.3
    ./subscriber A & local a=$!
    ./subscriber B & local b=$!

    local rc=0
    wait $a || rc=$?
    wait $b || rc=$?
    wait $owner || rc=$?
    echo "exit rc=$rc"
    return $rc
}

run_demos 2>&1 | tee "$OUT"
