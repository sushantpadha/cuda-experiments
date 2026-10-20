# libvmem test results

These are the measured results of the tests described in [README.md](README.md). All runs are on an RTX 4050 Laptop (5770 MiB of VRAM) with 15 GB of RAM and CUDA 13.0. Each configuration ran once.

## Findings

1. **Device memory from VMM is as fast as `cudaMalloc`.** When the data fits, the sharpen step takes 0.92 s on `vmem` and 1.04 s on `cuda`. Managed memory (`raw`, `uvm`) takes about five times longer, because every page faults on first touch.
2. **When the data does not fit, `vmem` gets through fastest.** `cuda` runs out of memory in every such run. With four processes sharing the GPU, the image test takes 2.3 s on `vmem` with the daemon, 3.3 s on `vmem` alone and 5.3 s on `uvm`.
3. **The price is pinned host memory, and the daemon caps it.** On its own, `vmem` accepts every eviction and DRAM use climbs to 5.9 GB. With `vmemd`, the 3000 MiB host budget holds DRAM at 3.4 GB. The daemon refuses the hints that would go over the budget, and those images stay on the device.
4. **Sharing the GPU through the daemon does not change results.** In every four-process run on `vmem` with `vmemd`, every worker passed its spot checks, and the daemon's counts returned to zero when the workers exited.

## Four processes at once (`make trace-all`)

These traces are from 2026-10-21. Each plot overlays one run per backend. The left axis shows VRAM (solid) and DRAM (dashed) above their values at the start, with a dotted line at each peak. The right axis counts the hints each `vmem` run had accepted so far. `cuda` is left out because it ran out of memory in both workloads. The CSVs, `cuda` included, are in [`analysis/`](../analysis/).

![image test, memory over time](../analysis/image.png)

| Image test | Time | Peak VRAM | Peak DRAM | Hints accepted |
|---|---|---|---|---|
| `uvm` | 5.3 s | 4771 MiB | 3783 MiB | 72 of 72 |
| `vmem` local | 3.3 s | 1714 MiB | 5882 MiB | 72 of 72 |
| `vmem` + `vmemd` | 2.3 s | 4206 MiB | 3409 MiB | 54 of 72 |

![GEMM test, memory over time](../analysis/gemm.png)

| GEMM test | Time | Peak VRAM | Peak DRAM |
|---|---|---|---|
| `uvm` | 16.7 s | 5224 MiB | 365 MiB |
| `vmem` local | 15.7 s | 5626 MiB | 1221 MiB |
| `vmem` + `vmemd` | 15.8 s | 5050 MiB | 1737 MiB |

The GEMM test makes no hints, so its matrices stay where they were placed. Without the daemon, `vmem` fills the device and puts 1.2 GB on the host. With the daemon, its 5400 MiB device limit moves about 0.5 GB more to the host, at no cost in time. The workers finish one after another, smallest matrix first, because without MPS their kernels take turns on the GPU.

## One process (`make stress`)

These runs are from 2026-10-20. Each used four images of 2404 MiB and 8192 × 8192 matrices, and every run gave the checksum `12b62e4bb28`. With `--resident` the images need more than the VRAM.

| Mode | Total | Sharpen | Peak VRAM | Peak DRAM |
|---|---|---|---|---|
| raw | 6.99 s | 5.61 s | 4916 MiB | 146 MiB |
| cuda | 2.31 s | 1.04 s | 4814 MiB | 38 MiB |
| uvm | 6.15 s | 4.79 s | 4914 MiB | 51 MiB |
| vmem | 2.16 s | 0.92 s | 4814 MiB | 10 MiB |
| raw `--resident` | 41.21 s | 39.82 s | 5666 MiB | 5710 MiB |
| uvm `--resident` | 14.18 s | 12.78 s | 5666 MiB | 9320 MiB |
| vmem `--resident` | 9.60 s | 8.36 s | 4830 MiB | 7264 MiB |
| vmem `--resident`, with `vmemd` | 11.41 s | 10.18 s | 4828 MiB | 7015 MiB |
| cuda `--resident` | out of memory at image 2 | | | |

## Caveats

Every number comes from a single run. VRAM and DRAM cover the whole GPU and the whole machine, so other programs shift them. We have not measured how much time each daemon round trip costs. In the stress test the daemon made `--resident` slower (11.4 s against 9.6 s), while in the four-process image test it made it faster. One likely reason is that refused evictions skip copies, but we have not confirmed it.
