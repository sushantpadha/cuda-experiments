# libvmem test results

These are the measured results of the tests described in [README.md](README.md). All runs are on an RTX 4050 Laptop (5770 MiB of VRAM) with 15 GB of RAM and CUDA 13.0. Each configuration ran once.

## Findings

1. **Device memory from VMM is as fast as `cudaMalloc`.** When the data fits, the sharpen step takes 0.92 s on `vmem` and 1.04 s on `cuda`. Managed memory (`raw`, `uvm`) takes about five times longer, because every page faults on first touch.
2. **When the data does not fit, `vmem` gets through fastest.** `cuda` runs out of memory in every such run. With four processes sharing the GPU, the image test takes 2.2 s on `vmem` with the daemon, 2.9 s on `vmem` alone and 5.3 s on `uvm`.
3. **The price is pinned host memory, and the daemon caps it.** On its own, `vmem` carries out every hint, 72 moves in all, and DRAM use climbs to 5.8 GB. With `vmemd`, the 3000 MiB host budget holds DRAM at 3.4 GB. The daemon refuses the 17 evictions that would go over the budget, and those images stay on the device, so only 46 hints move data. The 26 copies it saves are why the daemon run is faster here.
4. **Sharing the GPU through the daemon does not change results.** In every four-process run on `vmem` with `vmemd`, every worker passed its spot checks, and the daemon's counts returned to zero when the workers exited.

## Four processes at once (`make trace-all`)

These traces are from 2026-10-21, one run per backend, before the image test got its current sizes: each worker then made six images of 128 to 320 MiB (5.4 GB in all) and went over them once. The `raw` baseline and the mix test did not exist yet. Each plot overlays one run per backend. The left axis shows VRAM (solid) and DRAM (dashed) above their values at the start, with a dotted line at each peak. The right axis counts the hints that moved data in each `vmem` run, each step at the time the hint returned. A hint that is accepted but finds the image already in place (a prefetch after a refused eviction) moves nothing and is not counted. `cuda` is left out because it ran out of memory in both workloads. The CSVs and one plot per run, `cuda` included, are in [`analysis/`](../analysis/) as `WORKLOAD-BACKEND.csv`, `.events.csv` (one row per hint) and `.png`.

![image test, memory over time](../analysis/image.png)

| Image test | Time | Peak VRAM | Peak DRAM | Hints that moved data | Already in place | Refused |
|---|---|---|---|---|---|---|
| `uvm` | 5.3 s | 5663 MiB | 4251 MiB | 72 | 0 | 0 |
| `vmem` local | 2.9 s | 1586 MiB | 5839 MiB | 72 | 0 | 0 |
| `vmem` + `vmemd` | 2.2 s | 4014 MiB | 3420 MiB | 46 | 9 | 17 |

Until the host budget fills at about 0.86 s, the two `vmem` runs move images at the same rate. From then on the daemon refuses evictions, and its run stops at 46 moves while the local run goes on to 72.

![GEMM test, memory over time](../analysis/gemm.png)

| GEMM test | Time | Peak VRAM | Peak DRAM |
|---|---|---|---|
| `uvm` | 16.7 s | 5224 MiB | 339 MiB |
| `vmem` local | 15.8 s | 5626 MiB | 1124 MiB |
| `vmem` + `vmemd` | 15.8 s | 5226 MiB | 1507 MiB |

The GEMM test makes no hints, so its matrices stay where they were placed. Without the daemon, `vmem` fills the device and puts 1.1 GB on the host. With the daemon, its 5400 MiB device limit moves about 0.4 GB more to the host, at no cost in time. The workers finish one after another, smallest matrix first, because without MPS their kernels take turns on the GPU.

## One process

These runs are from 2026-10-20, made with the single-process `stress` program that the churn stress test has since replaced. It ran a matrix multiply and then the image workload; the image part now runs as `make test-workers-image ARGS="--workers 1 --image-mib 2404 --images 4 --passes 1 --backend BACKEND"`, with `--no-resident` for the rows without `--resident`. Each used four images of 2404 MiB and 8192 × 8192 matrices, and every run gave the checksum `12b62e4bb28`. With `--resident` the images need more than the VRAM.

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

Every number comes from a single run. VRAM and DRAM cover the whole GPU and the whole machine, so other programs shift them. We have not measured how much time each daemon round trip costs. In the single-process stress test the daemon made `--resident` slower (11.4 s against 9.6 s), even though it also refused hints there; that run had no hint timing, so we cannot yet say where the time went.
