# libvmem: VMM-based Generalized GPU Memory Virtualization for Multi-Tenant applications

[**Report (PDF)**](report/report.pdf) | [**Tracker**](TRACKER.md) | [v0 primitives](primitives/) | [VMM latency](experiments/vmm-latency/) | [notes](notes/)

**Authors:** Sushant Padha (24B1057), Koduru Tejeswar (24B0918)
**Mentor:** Prof. Purushottam Kulkarni
**Department:** Computer Science and Engineering, IIT Bombay

## Idea

GPU memory placement is decided by the driver: with Unified Memory (UVM) pages migrate on page faults under one device-wide policy; with `cudaMalloc` everything is manual. CUDA's Virtual Memory Management (VMM) API lets a program reserve addresses, choose where each piece of physical memory lives (VRAM or pinned host memory), and remap it at the same address.

`libvmem` shares one GPU between several processes **at the same time**. Nixie (OSDI '26) does it one application at a time. Apps call a slab allocator directly. A daemon, `vmemd`, decides if each new chunk goes to VRAM or to pinned host memory (read by kernels over PCIe). Memory in use never moves, because moving it under a running kernel crashes or loses writes. Apps can hint `evict`/`prefetch` on large allocations. Userspace only. Goals are provisional.

## Plan

Built in stages (tracker goal 1):

1. **v0 primitives** (done): a per-process wrapper over VMM and an action runner, in [`primitives/`](primitives/). They showed that memory cannot be remapped under a running kernel.
2. **libvmem + vmemd** (current): apps call the allocator, the daemon places chunks, and clients run together under MPS. Later: policies and per-block content hashing. Code in [`libvmem/`](libvmem/).

Alongside: a UVM vs. VMM study and comparison with Nixie, MSched and tuned UVM (tracker goal 5), and the report.

## Where things stand

| Area | State | Look at |
|---|---|---|
| v0 primitives | done; showed that live remap is impossible in userspace | [`primitives/`](primitives/) |
| libvmem + vmemd | `cuda` and `uvm` backends done; `vmem` backend next, then `vmemd`, then MPS | [`libvmem/`](libvmem/) |
| VMM call latency | done, compared with `cudaMalloc` and `cudaMallocManaged` | [`experiments/vmm-latency/`](experiments/vmm-latency/) |
| Compute sharing, eviction signals | done | [`notes/compute-sharing.md`](notes/compute-sharing.md), [`notes/eviction-signals.md`](notes/eviction-signals.md), [`experiments/lookups/`](experiments/lookups/) |
| Related work | Nixie, Prism read; MSched next | [`notes/nixie.md`](notes/nixie.md), [`notes/prism.md`](notes/prism.md) |
| Report | outline is stale since the pivot; UVM vs. VMM section drafted; latency section stubbed | [`report/`](report/) |

Full goals and status: [`TRACKER.md`](TRACKER.md).

## Key observations so far

Preliminary, single machine (RTX 4050 Laptop, 6 GB, CUDA 13.0), 1 to 3 runs each.

- Without MPS, kernels from different processes take turns on the GPU; with MPS they run at once. Under MPS one client's bad memory access kills every client, so evicted memory must always stay mapped.
- Kernels read host-backed VMM memory at about 13 GB/s (device: about 187 GB/s). Remapping a buffer while a kernel reads it crashes; remapping while unrelated kernels run is fine.
- Device VMM calls mostly cost a fixed 3 to 230 µs whatever the size (release grows slowly); host-backed `cuMemCreate` costs about 115 µs per MiB. Back-to-back device mappings sometimes stall about 2 ms.
- A CUPTI-injected scan of kernel launch arguments finds which buffers a kernel may touch, library kernels included, at negligible cost.

## Repository layout

```
primitives/    libvmem v0: src/ (vmem.cuh, action runner), actions/ (.vm files), tests/
libvmem/       per-process allocator over a backend (vmem, uvm, cuda); src/libvmem.cuh, src/backends/, tests/
experiments/   scratch experiments: vmm-latency, lookups (MPS, eviction signals),
               VMMVector, VMMRemapShared, VMMSlab, pytorch-vmm-study, warmups, dbg
notes/         short notes on papers and findings
report/        LaTeX report; make -> report/report.pdf
references/    links to the literature and documentation used
TRACKER.md     goals, status, current stage
CLAUDE.md      instructions for AI coding agents working in the repository
```

`main` is the published branch.

## Building

Needs an NVIDIA GPU with VMM support, CUDA 13.0 and the driver API library.

```
cd primitives && make test N=1               # v0 primitives test; make run N=1, make fuzz
cd libvmem && make test N=1c                 # allocator tests: 1c 1u 2c 2u
cd experiments/vmm-latency && ./run.sh       # latency experiment (needs nsys, python3, matplotlib)
cd report && make                            # report (pdflatex, bibtex)
```

Everything under `experiments/` is exploratory: results are preliminary observations from one machine and may change.
