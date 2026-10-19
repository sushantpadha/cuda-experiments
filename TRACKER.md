# Tracker

**libvmem**: VMM-based Generalized GPU Memory Virtualization for Multi-Tenant applications. Several processes share one GPU at the same time. Apps call a slab allocator directly; a daemon (`vmemd`) picks VRAM or pinned host memory for each new chunk. Memory in use never moves.
Sushant Padha (S), Koduru Tejeswar (K). Mentor: Prof. Purushottam Kulkarni. IIT Bombay, CSE.

This file is the authoritative source for goals, subgoals, priorities and status. Other docs link here by ID.
Status: `[ ]` pending, `[~]` in progress, `[x]` done (remove and move to Log). Goals are numbered 1, 2; subgoals 1A, 1B; sub-subgoals 1Ai, 1Aii.
Timeline: full project about 1.5 months. No code deadline.
Current stage: 1B vmemd + slab. Next: 1Biv.

## 1. Prototype (top priority)

Order: v0 primitives, then vmemd + slab, then compute sharing and policies. Correctness first, then reasonable efficiency.

- [~] 1A v0 primitives on VMM (`primitives/`): Manager, runner (1Ai), tests, fuzz done. Left: fix `vmem.cuh` (remap rollback, `map` cleanup). `tests/test4.cu`: remap under a running kernel crashes or loses writes
  - [ ] 1Aii Run many in parallel; check correctness and speed
- [ ] 1B vmemd + slab: apps call `libvmem` directly; `vmemd` picks device or host for each new chunk; memory in use never moves
  - [ ] 1Bi Design left: chunk size, client-daemon protocol, device budget (count per-process context overhead)
  - [~] 1Bii Allocator in local mode (no daemon). Done with `cuda` and `uvm` backends; `vmem` backend left
  - [ ] 1Biii `vmemd` broker: per-tenant usage, device budget. Rule: device until full, then host. Client holds its memory; client crash = socket close; daemon down = local mode
  - [ ] 1Biv (next) `evict`/`prefetch` on the `vmem` backend. Later: non-blocking version; the host must run the `cuMemUnmap` + `cuMemMap` swap, and no event can order it
  - [ ] 1Bv Test apps, multi-process runs, `vmem`/`uvm`/`cuda` baselines
- [ ] 1C True compute sharing: MPS and/or green contexts (2A)
- [ ] 1D Smarter `vmemd` policies: quotas, fairness
- [ ] 1E Per-block content hashing: share identical blocks across processes

## 2. Look-ups

- [~] 2A MPS and Green Contexts for true spatial compute sharing; check that VMM calls and handle export work under them. First pass done: `notes/compute-sharing.md`
- [~] 2B Ways to get metrics for an eviction policy (no hardware access bits for VMM ranges; launch arguments, kernel times, hashing, one-off offline profiling). First pass done: `notes/eviction-signals.md`
- [ ] 2C Scheduling and eviction policies to borrow: memory tiering (TPP, HeMem, Memtis), caching (ARC, GreedyDual), ESX shares and idle tax, BoxD limits

## 3. Use cases

- [ ] 3A LLM inference: hints on KV cache vs. model weight allocations
- [ ] 3B Streaming workloads (batch image processing, data larger than VRAM)

## 4. Related work

- [ ] 4A MSched (arXiv 2512.24637): read in full
- [ ] 4B General background: GMLake, vAttention, vTensor, Concordia
- [ ] 4C Ask Soham for verified page-fault cost numbers and papers
- [ ] 4D Verify or drop the remaining unverified related-work entries

## 5. UVM vs. VMM study and comparison

- [ ] 5B Fault-driven paging vs. prefetch and remap. Baselines: plain UVM and UVM with advise and prefetch, always both
- [ ] 5C Further UVM vs. VMM: thrash under oversubscription, first-touch cost, UVM with and without advise, host memory kept behind GPU-resident UVM pages
- [ ] 5D Read the UVM driver (driver 580): `uvm_pmm_gpu.c`, `uvm_gpu_access_counters.c`, `uvm_perf_thrashing.c`
- [ ] 5E Side-by-side runs on this machine, same workloads: Nixie, plain UVM, UVM with advise and prefetch, nvshare, MSched if released, kvcached, manual `cudaMalloc`
- [~] 5F Learn Nsight Systems (in progress) and Nsight Compute; baseline traces

## 6. PyTorch

- [ ] 6A Caching allocator integration (later). Study done on branch `pytorch-study`; no further study planned

## 7. Report (`report/`)

- [ ] 7A Introduction
- [ ] 7B PyTorch allocator section, from the existing study
- [ ] 7C Design section. Follows 1A to 1D
- [ ] 7D Experiments section. Needs 1A, 5B, 5E
- [ ] 7E Abstract and conclusion. Last
- [ ] 7F UVM vs. VMM section: drafted, needs 5B results
- [ ] 7G Related work section (Nixie, Prism, MSched, BoxD, GMLake, MPS and green contexts). Needs 4A
- [ ] 7H VMM call latency section (stub in place). Source: `experiments/vmm-latency/README.md`

## 8. Housekeeping

- [ ] 8A Purge tracked PDFs and tool state from git history and force-push `main` (S). Needs explicit permission to rewrite history

## Log

- 2026-09-23: tracker and report skeleton created (project then named `vmadvise`).
- 2026-09-24: repo reorganised; PyTorch allocator study done on `pytorch-study`; BoxD read.
- 2026-09-27: notes moved to `notes/`.
- 2026-09-28: Nixie and MSched cover most of the original pitch.
- 2026-09-29: direction set: Nixie-style multiplexing with spatial sharing, host fallback, general workloads. PyTorch study ended.
- 2026-10-04: renamed `libvmem`; work moved to `main`.
- 2026-10-05: Nixie and Prism read. 2A/2B first pass. 1A primitives written. 5A first run.
- 2026-10-07: 1Ai done: runner, tests, fuzz (0 disagreements in 600 programs).
- 2026-10-09: 5A done: VMM matches `cudaMalloc`; managed cost moves to first touch. A daemon hides call costs.
- 2026-10-14: `test4.cu`: remap under a running kernel crashes or loses writes. Dropped live remap, launch interception, `LD_PRELOAD`. New core: vmemd + slab.
- 2026-10-19: `libvmem/` started: backend interface, SmallPool and LargePool. `evict`/`prefetch` move a large chunk now; `free` is a plain free.
- 2026-10-20: `cuda` and `uvm` backends done, tests 1 and 2 pass on both.
