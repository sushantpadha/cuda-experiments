# VMM call latency

How long each CUDA VMM call takes on this machine, for 2 MiB to 1 GiB, device vs. pinned host backing, read/write vs. read-only access. Tracker 5 (UVM vs. VMM study).

```
./run.sh        # build, trace with nsys, export, analyse (about 40 s); ./run.sh 50 for more reps
```

Each repetition runs `reserve, create, map, set access, unmap, release, address free` on a fresh allocation, inside one NVTX range. nsys traces the driver API; `analyze.py` matches every call to its range. All configurations run in one shuffled order (fixed seed), 20 reps each after 3 warm-up reps, in two modes:

- **spaced:** 20 ms busy-wait between reps, so the driver's background work from the previous free has finished.
- **back to back:** no gap.

Outputs in `results/`: `latency.csv` (median, quartiles, max per mode, location, size, access, call), `vmm_latency.png`, `vmm_backtoback.png`. Published numbers for comparison: `published.md`.

![latency](results/vmm_latency.png)
![back to back](results/vmm_backtoback.png)

## Results (spaced, read/write, median µs)

| Size | Device: create | map | set access | unmap | release | Host: create | set access | release |
|---|---|---|---|---|---|---|---|---|
| 2 MiB | 57 | 2.7 | 56 | 30 | 24 | 297 | 100 | 43 |
| 32 MiB | 70 | 2.8 | 96 | 36 | 36 | 4430 | 399 | 467 |
| 1 GiB | 68 | 1.6 | 98 | 34 | 388 | 91361 | 8737 | 24611 |

- **Device memory: nearly flat in size.** Create, map, set access and unmap stay within 2 to 100 µs from 2 MiB to 1 GiB; only release grows (24 to 388 µs).
- **Host memory: create grows linearly**, about 89 µs per MiB (1 GiB takes 91 ms), as do set access (about 8.5 µs/MiB) and release (about 24 µs/MiB). So moving a buffer to host costs about as much to allocate its host backing (91 ms per GiB) as to copy it (about 80 ms per GiB at 13 GB/s).
- **Read-only vs. read/write access costs the same** within run-to-run noise (within 25% at every size).
- **Map is cheap everywhere** (2 to 10 µs); the cost of making memory usable is in create and set access.
- **Back to back, device set access and unmap stall about 2 ms** on roughly a third of calls (40 of 120 and 42 of 120 above 1 ms, vs. 2 and 3 spaced), at every size. The stall goes away with a 20 ms gap, so it is most likely driver background work left over from earlier frees, not the cost of the call itself.
- Compared with vAttention's 2 MB numbers (create 29, set access 38, unmap 34 µs; GPU not stated), our device calls are within about 2x.

## Caveats

- One machine, one run of 20 reps per point. Device medians varied noticeably between trial runs, because of the bimodal stalls.
- Spacing calls changes CPU and GPU power state: a 20 ms *sleep* made even the CPU-only `cuMemMap` up to about 15x slower, so the gap is a busy-wait. Real latency depends on what ran just before.
- nsys measures host-side call duration.
