# vmadvise: A Userspace Residency-Hinting Library for GPU Memory

[**Report (PDF)**](report/report.pdf) | [**Tracker**](TRACKER.md) | [nixie notes](notes/nixie.md)

**Authors:** Sushant Padha (24B1057), Koduru Tejeswar (24B0918)
**Mentor:** Prof. Purushottam Kulkarni
**Department:** Computer Science and Engineering, IIT Bombay

## Overview

GPU memory placement is largely decided by the driver. With CUDA Unified Memory (UVM), pages migrate on demand and residency is governed by device-wide driver policy; with `cudaMalloc`, data movement is entirely manual. CUDA's Virtual Memory Management (VMM) API offers a third option: an application can reserve address space, choose where each piece of physical memory lives (device or host), and remap it at will.

This project studies how far that control can be turned into a small userspace library, `vmadvise`, in the spirit of `madvise`: applications state residency hints and priorities, and the library realises them with VMM, without driver changes. Target workloads are streaming and batch-style GPU programs, such as image processing, and multi-process sharing of one GPU. The goals are provisional and will be refined as the project develops.

## Status

Goals, subgoals and status live in [`TRACKER.md`](TRACKER.md). Summary:

1. **Read related work (top priority):** Nixie and GMLake papers first, then vAttention and vTensor; deeper contrast with MSched, Prism, kvcached.
2. **Design:** five gated phases (use cases, requirements, features, API, design). Now in phase 1. Candidate features: per-block content hashing (skip unchanged copies, cross-process sharing, copy-before-write) and `madvise`-style hints.
3. **UVM vs. VMM experiments:** fault-driven paging against prefetch and remap, plus side-by-side runs against Nixie, tuned UVM and other systems. Pending.
4. **Profiling and prototype:** Nsight, baseline traces, userspace allocator. Pending.
5. **PyTorch allocator study:** source study done on branch `pytorch-study`; hands-on runs pending.
6. **Report:** skeleton, build and UVM vs. VMM section written; the rest pending.
7. **Housekeeping.**

## Repository layout

```
report/        LaTeX report; make -> report/report.pdf
TRACKER.md     goals, status, and current phase
notes/         short notes; notes/artifacts/ has long-form research
experiments/   exploratory experiments (VMMVector, VMMRemapShared, VMMSlab,
               pytorch-vmm-study, warmups, dbg)
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

Everything under `experiments/` and `warmups/` is exploratory. Results in these directories are preliminary observations from a single machine; they may be revised or removed and should not be read as established findings.
