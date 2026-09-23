# vmadvise: A Userspace Residency-Hinting Library for GPU Memory

[**Report (PDF)**](report/report.pdf)

This repository holds the research and development work for `vmadvise`, a
userspace library for GPU memory management built on CUDA's Virtual Memory
Management (VMM) API. The goal is to provide `madvise`-style control over
CPU-GPU residency (priorities, sharing, and streaming-workload
optimizations) without writing driver-level code.

**Authors:** Sushant Padha (24B1057), Koduru Tejeswar (24B0918)
**Mentor:** Prof. Purushottam Kulkarni
**Department:** Computer Science and Engineering, IIT Bombay

## Report

The report source is in `report/`. Running `make` in that directory builds
`report/report.pdf`, which is committed alongside the source. Progress on
each part of the project is tracked in `TRACKER.md`.

## Status

| Work item | Status | Location |
|---|---|---|
| Written report | In progress. Section stubs exist; the UVM vs. VMM section is drafted. | `report/` |
| Basic VMM API demonstration | Done: a growable GPU vector built on `cuMemCreate`, `cuMemMap`, and `cuMemSetAccess`. | `VMMVector/` |
| PyTorch caching-allocator study | Not started. Reading notes only. | `pytorch-vmm-study/` |
| UVM vs. VMM experiments | Pending. | none yet |
| Feature list, requirements, and workloads | Pending. This includes workloads that benefit from the library and workloads that do not. | none yet |
| Nsight tracing and profiling | Pending. Learning the tools is the first step. | none yet |
| Userspace slab allocator on the VMM API | Pending. Depends on the feature list being settled first. | none yet |

## Repository layout

`CLAUDE.md` describes each directory, the hardware and toolchain, build
instructions, and code conventions. Most directories beyond `report/` contain
exploratory experiments that may later be revised or removed.
