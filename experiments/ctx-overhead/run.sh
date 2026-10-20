#!/bin/bash
# runs the probe lazy, eager, and under MPS (starts and stops its own MPS daemon); writes results/ctx_overhead.csv
set -e
cd "$(dirname "$0")" && make -s
mkdir -p results
out=results/ctx_overhead.csv
echo "mode,rep,step,delta_mib" > $out
./ctx_overhead 5 | tail -n +3 | sed 's/^/lazy,/' >> $out
CUDA_MODULE_LOADING=EAGER ./ctx_overhead 5 | tail -n +3 | sed 's/^/eager,/' >> $out
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-ctx CUDA_MPS_LOG_DIRECTORY=/tmp/mps-ctx-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
./ctx_overhead 5 | tail -n +3 | sed 's/^/mps,/' >> $out
echo quit | nvidia-cuda-mps-control
cat $out
