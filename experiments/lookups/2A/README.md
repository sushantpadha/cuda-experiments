# 2A probes: compute sharing and VMM under MPS / green contexts

Scratch.

Build: `nvcc -O2 -std=c++17 -arch=sm_89 spatial.cu -o spatial -lcuda`

Modes:

```
./spatial work <iters> <blocks> [green_sms]   # fixed-work kernel; prints device-clock interval and SM ids
./spatial green <iters> <blocks> <sms_a>      # two green contexts in one process
./spatial vmm                                 # device + HOST_NUMA backing, kernel R/W, remap, bandwidth
./spatial share                               # export/import a VMM handle across a fork (device and host)
./spatial remapbusy <iters>                   # remap one buffer while an unrelated kernel runs
./spatial fault                               # kernel touches an unmapped VMM range
./spatial hold <secs>                         # hold a context (per-context VRAM overhead)
./spatial hostalloc <MiB>                     # HOST_NUMA-only cuMemCreate (MPS cap check)
./spatial idlecheck <secs>                    # idle, then launch (does a co-tenant fault hit idle clients?)
./spatial gaprace                             # remap a buffer while a kernel reads it
```

`green_sms < 0` uses the SMs left over after splitting off `|green_sms|`, so two processes get disjoint groups.

## Runs

Without MPS:

```
./spatial work 250000000 4                                     # alone, ~450 ms
(./spatial work 250000000 4 & ./spatial work 250000000 4 & wait) # ~915 ms each: time-sliced
./spatial green 250000000 40 10                                # disjoint SM sets, concurrent
(./spatial work 250000000 40 10 & ./spatial work 250000000 40 -10 & wait)  # still time-sliced
./spatial vmm; ./spatial share; ./spatial remapbusy 250000000
./spatial gaprace                                              # remap under a reading kernel -> IllegalAddress
(./spatial work 1600000000 4 & sleep 1.5; ./spatial fault & wait)          # B survives
```

With MPS (user-local daemon; the pipe path must be short, unix sockets cap at 108 chars):

```
mkdir -p /tmp/mpsp /tmp/mpsl
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mpsp CUDA_MPS_LOG_DIRECTORY=/tmp/mpsl
nvidia-cuda-mps-control -d
(./spatial work 250000000 4 & ./spatial work 250000000 4 & wait) # ~445 ms each: concurrent
./spatial vmm; ./spatial share
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=128M" ./spatial vmm         # cuMemCreate -> OUT_OF_MEMORY
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25 ./spatial work 250000000 40 # 4 SMs, ~1500 ms
(./spatial work 1600000000 4 & sleep 1.5; ./spatial fault & wait) # B also fails: IllegalAddress
grep "connected" /tmp/mpsl/server.log | tail                      # confirms clients went through MPS
(./spatial work 250000000 40 10 & ./spatial work 250000000 40 10 & wait)  # same green group: ~660 / ~1190 ms
(./spatial work 250000000 40 10 & ./spatial work 250000000 40 -10 & wait) # disjoint groups: ~630 / ~630 ms
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=128M" ./spatial hostalloc 256    # succeeds: cap ignores HOST_NUMA
(./spatial idlecheck 4 & sleep 1.5; ./spatial fault & wait)          # idle client also gets IllegalAddress
(./spatial vmm > a.txt & ./spatial vmm > b.txt & wait); grep remap a.txt b.txt   # shared-PCIe copies
for i in 1 2 3 4; do ./spatial hold 30 >/dev/null & sleep 3; nvidia-smi --query-gpu=memory.used --format=csv,noheader; done; wait
echo quit | nvidia-cuda-mps-control                              # stop the daemon
```
