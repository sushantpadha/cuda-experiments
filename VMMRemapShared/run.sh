#!/bin/bash
# Build + run both demos. Usage: ./run.sh
set -e
cd "$(dirname "$0")"
make -s

echo "======== remap demo ========"
./remap_test "${1:-1048576}" "${2:-2}"

echo
echo "======== multi-process subscribe demo ========"
rm -f /dev/shm/vmm_share_state /tmp/vmm_share.sock
./owner 2 4 2 &
OWNER=$!
sleep 0.3
./subscriber A &
A=$!
./subscriber B &
B=$!

rc=0
wait $A || rc=$?
wait $B || rc=$?
wait $OWNER || rc=$?
echo "exit: owner+subs rc=$rc"
exit $rc
