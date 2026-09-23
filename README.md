# vmadvise: A Userspace Residency-Hinting Library for GPU Memory

[**Report (PDF)**](report/report.pdf)

**Authors:** Sushant Padha (24B1057), Koduru Tejeswar (24B0918)
**Mentor:** Prof. Purushottam Kulkarni
**Department:** Computer Science and Engineering, IIT Bombay

## Overview

GPU memory placement is largely decided by the driver. With CUDA Unified Memory (UVM), pages migrate on demand and residency is governed by device-wide driver policy; with `cudaMalloc`, data movement is entirely manual. CUDA's Virtual Memory Management (VMM) API offers a third option: an application can reserve address space, choose where each piece of physical memory lives (device or host), and remap it at will.

This project studies how far that control can be turned into a small userspace library, `vmadvise`, in the spirit of `madvise`: applications state residency hints and priorities, and the library realises them with VMM, without driver changes. Target workloads are streaming and batch-style GPU programs, such as image processing, and multi-process sharing of one GPU. The goals are provisional and will be refined as the project develops.

## Status

- **Report.** The skeleton, build, and the UVM vs. VMM comparison are written. The introduction, allocator study, design, experiments, and conclusion are pending.
- **Design.** Use cases, requirements, and features are still to be settled. The API and the design follow from them.
- **VMM demonstrations.** A growable GPU vector on VMM (`experiments/VMMVector`) and a prototype of device-to-host remapping with a multi-process allocator (`experiments/VMMRemapShared`). Both are exploratory.
- **PyTorch caching-allocator study.** Source notes and preliminary experiments exist on the local branch `pytorch-study`. Hands-on runs with the memory visualizer are pending.
- **UVM vs. VMM experiments.** Pending.
- **Nsight profiling and tracing.** Pending.
- **Userspace allocator.** Pending; it depends on the feature list.

Detailed, item-level progress is in [`TRACKER.md`](TRACKER.md).

## Repository layout

```
report/        LaTeX report; make -> report/report.pdf
TRACKER.md     goals, status, and current phase
experiments/   exploratory experiments (VMMVector, VMMRemapShared, pytorch-vmm-study)
warmups/       small standalone kernels and device probes
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
