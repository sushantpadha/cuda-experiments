# CUDA context overhead

This probe measures how much VRAM one process's CUDA context takes outside any allocator, so that `vmemd` can count it against a tenant's device usage (tracker 1Bii).

```
./run.sh                                   # lazy, eager, and MPS runs; writes results/ctx_overhead.csv (about 1 min)
./ctx_overhead 5                           # one mode by hand; CUDA_MODULE_LOADING=EAGER ./ctx_overhead 5 for eager
```

## Method

The parent process reads device-wide used VRAM through NVML, which does not create a CUDA context. It forks a child and steps it through four stages over a pipe: `cudaFree(0)`, a first kernel (after a 4-byte `cudaMalloc`), `cublasCreate`, and a 64x64 `cublasSgemm`. After each stage the parent waits 200 ms and reads NVML again. The child stays alive until the parent has read the last stage, and the parent reads once more after the child exits. Each mode runs 5 times in fresh processes. `run.sh` starts its own MPS control daemon for the MPS mode and stops it at the end.

## Results

We ran this on 2026-10-20 on the RTX 4050 Laptop (driver 580, CUDA 13.0). Every repetition gave the same numbers. The table shows VRAM used above the baseline, in MiB.

| Stage | Lazy (default) | Eager | MPS client |
|---|---|---|---|
| `cudaFree(0)` | 89.9 | 89.9 | 80.6 |
| first kernel | 91.9 | 91.9 | 82.6 |
| `cublasCreate` | 101.9 | 281.9 | 92.6 |
| first `sgemm` | 101.9 | 305.9 | 92.6 |
| after exit | 0.0 | 0.0 | 0.0 |

A bare context takes 90 MiB. The 2 MiB step at the first kernel comes from the `cudaMalloc` page, not from the kernel. With lazy loading (the default), cuBLAS adds 10 MiB; with `CUDA_MODULE_LOADING=EAGER` it adds 190 MiB at `cublasCreate` and 24 MiB more at the first `sgemm`. Under MPS each client takes 81 MiB, and the MPS server holds another 37 MiB once, which stays after the first client exits.

We set `ctx_overhead_mib = 92` in `libvmem/daemon/vmemd.toml`: a bare context plus its first page. These numbers are lower bounds. Kernels load lazily, so an app that uses more libraries or more kernels takes more VRAM as it runs.
