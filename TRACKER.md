# Tracker

**libvmem**: VMM-based Generalized GPU Memory Virtualization for Multi-Tenant applications. Nixie-style GPU memory multiplexing on VMM, but with true spatial sharing, a pinned host-memory fallback instead of faults, general (non-LLM) workloads, and optional `madvise`-style hints.
Sushant Padha (S), Koduru Tejeswar (K). Mentor: Prof. Purushottam Kulkarni. IIT Bombay, CSE.

This file is the authoritative source for goals, subgoals, priorities and status. Other docs link here by ID.
Status: `[ ]` pending, `[~]` in progress, `[x]` done (remove and move to Log). Goals are numbered 1, 2; subgoals 1A, 1B; sub-subgoals 1Ai, 1Aii.
Timeline: full project about 1.5 months. No code deadline.
Current stage: prototype v0.

## 1. Prototype (top priority)

Order: primitives first (as in Nixie), then transparent integration, then true compute sharing, then features. Correctness and basic speed are checked at each step.

- [~] 1A v0 primitives on VMM (`primitives/vmem.cuh`): written, smoke test passes. Single-threaded; mutex-protected version later
  - [ ] 1Ai Action runner: reads a simple action file and executes it
  - [ ] 1Aii Run many in parallel; check correctness and basic speed
- [ ] 1B v1 transparent integration: hook any `cudaMalloc`-based program (CUPTI injection or `LD_PRELOAD`); `remap` as a general hook, run on OOM or by basic scheduling; check correctness and basic speed
- [ ] 1C True compute sharing: run tenants under MPS and/or green contexts so kernels from different processes run at the same time; keep evicted ranges always mapped, since one MPS client's fault kills all clients (2A)
- [ ] 1D Hints and policies: `madvise`-style hints from the app, plus smarter eviction from and prefetching into VRAM driven by runtime signals (CUPTI launch-argument scan, kernel times; 2B)
- [ ] 1E Profile-guided hints: run the app once under a Compute Sanitizer (or NVBit) profiler, record which buffers each kernel touches, and let the daemon use that profile as automatic hints
- [ ] 1F Per-block content hashing: skip copying unchanged blocks, share identical blocks across processes, copy-before-write

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

- [~] 5A VMM call latency (higher priority) vs. size, device vs. host, access, spacing (`experiments/vmm-latency/`). First run done
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

## 8. Housekeeping

- [ ] 8A Purge tracked PDFs and tool state from git history and force-push `main` (S). Needs explicit permission to rewrite history

## Log

- 2026-09-23: tracker and report skeleton created (project then named `vmadvise`).
- 2026-09-24: repo reorganised; PyTorch allocator study done on branch `pytorch-study` (preliminary); BoxD read.
- 2026-09-27: notes moved to `notes/`.
- 2026-09-28: Nixie and MSched found to cover most of the original pitch.
- 2026-09-29: direction set: Nixie-style multiplexing with true spatial sharing, pinned host fallback, general workloads, optional hints. Design phases replaced by prototype stages. PyTorch study ended.
- 2026-10-04: renamed `libvmem`; work moved to `main`.
- 2026-10-05: Nixie and Prism read (Nixie: basic design to expand; Prism: shows possible impact, too LLM-specific to build on). 2A/2B first pass: MPS needed for cross-process concurrency, one MPS client's fault kills all clients, green contexts inside MPS clients isolate SMs, VMM works under MPS; CUPTI launch-argument scan sees library kernels, hardware access counters unusable. 1A primitives written. 5A first run: device VMM calls nearly flat in size, host create about 89 us/MiB, back-to-back device calls stall about 2 ms. Report outline matched to the plan. Preliminary, 1 to 3 runs.
