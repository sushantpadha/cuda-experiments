# libvmem: VMM-based Generalized GPU Memory Virtualization for Multi-Tenant applications

[**Report (PDF)**](report/report.pdf) | [**Tracker**](TRACKER.md) | [nixie notes](notes/nixie.md) | [prism notes](notes/prism.md)

**Authors:** Sushant Padha (24B1057), Koduru Tejeswar (24B0918)
**Mentor:** Prof. Purushottam Kulkarni
**Department:** Computer Science and Engineering, IIT Bombay

## Overview

GPU memory placement is largely decided by the driver. With CUDA Unified Memory (UVM), pages migrate on demand and residency is governed by device-wide driver policy; with `cudaMalloc`, data movement is entirely manual. CUDA's Virtual Memory Management (VMM) API offers a third option: an application can reserve address space, choose where each piece of physical memory lives (device or host), and remap it at will.

This project builds on that control to share one GPU's memory between several processes at once. Nixie (OSDI '26) multiplexes applications over time, one resident at a time; `libvmem` aims for spatial sharing, where applications run together and memory that does not fit in VRAM is remapped to pinned host memory, which kernels read over PCIe instead of faulting. It targets general CUDA workloads rather than LLM serving alone, and lets applications add optional `madvise`-style hints. The goals are provisional and will be refined as the project develops.

## Status

Goals, subgoals and status live in [`TRACKER.md`](TRACKER.md). Summary:

1. **Prototype (top priority):** v0 VMM primitives (alloc, map, unmap, remap, free) with an action runner run in parallel; v1 transparent hooks for `cudaMalloc` programs (CUPTI injection or `LD_PRELOAD`) with remap on OOM or basic scheduling; true compute sharing with MPS and/or green contexts; then hints and policies, profile-guided hints, content hashing.
2. **Look-ups:** MPS and Green Contexts for spatial compute sharing, and eviction signals: first pass done, in progress. Scheduling and eviction policies pending.
3. **Use cases:** LLM inference with hints on KV cache vs. weights; streaming workloads.
4. **Related work:** Nixie read (the basic design to start from and expand); Prism read (shows the impact possible, too LLM-specific to build on); MSched next.
5. **UVM vs. VMM study and comparison:** fault-driven paging vs. prefetch and remap, UVM driver internals, and side-by-side runs against Nixie, tuned UVM and other systems. Learning Nsight Systems in progress; the rest pending.
6. **PyTorch:** allocator study done; integration later.
7. **Report:** skeleton, build and UVM vs. VMM section written; the rest pending.
8. **Housekeeping.**

## Repository layout

```
report/        LaTeX report; make -> report/report.pdf
TRACKER.md     goals, status, and current stage
notes/         short notes; notes/artifacts/ has long-form research
experiments/   exploratory experiments (VMMVector, VMMRemapShared, VMMSlab,
               lookups, pytorch-vmm-study, warmups, dbg)
references/    links to the literature and documentation used
CLAUDE.md      instructions for AI coding agents working in the repository
```

`main` is the published branch. Further experiments live on other branches that are not published.

## Building

Report (needs `pdflatex` and `bibtex`):

```
cd report && make
```

Experiments (needs an NVIDIA GPU with VMM support, CUDA 13.0, and the driver API library):

```
cd experiments/VMMVector && make
./main 1000000 4 6
```

Code was developed on an RTX 4050 Laptop GPU (compute capability 8.9). Each experiment directory is self-contained.

## Status of the code

Everything under `experiments/` is exploratory. Results in these directories are preliminary observations from a single machine; they may be revised or removed and should not be read as established findings.
