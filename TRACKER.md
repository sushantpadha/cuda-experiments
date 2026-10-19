# Tracker

**libvmem**: VMM-based Generalized GPU Memory Virtualization for Multi-Tenant applications. Several processes share one GPU at the same time through a slab allocator on VMM: a daemon (`vmemd`) decides whether each new chunk goes to VRAM or pinned host memory, and memory in use never moves. Apps call the allocator directly (no interception), with optional reuse hints that libvmem acts on.
Sushant Padha (S), Koduru Tejeswar (K). Mentor: Prof. Purushottam Kulkarni. IIT Bombay, CSE.

This file is the authoritative source for goals, subgoals, priorities and status. Other docs link here by ID.
Status: `[ ]` pending, `[~]` in progress, `[x]` done (remove and move to Log). Goals are numbered 1, 2; subgoals 1A, 1B; sub-subgoals 1Ai, 1Aii.
Timeline: full project about 1.5 months. No code deadline.
Current stage: vmemd + slab (1B); v0 mostly done. Next: hint-driven moves (1Biv).

## 1. Prototype (top priority)

Order: v0 primitives, then vmemd + slab, then compute sharing and policies. Correctness first, then reasonable efficiency.

- [~] 1A v0 primitives on VMM (`primitives/`): Manager, action runner (1Ai), tests and fuzz test done. Left: fix `vmem.cuh` (remap releases old backing last and rolls back on failure; `map` unmaps if set access fails); sync before remap/unmap is the slab's job, which moves chunks only at full synchronization points. `tests/test4.cu` shows remap racing a running kernel crashes or loses writes
  - [ ] 1Aii Run many in parallel; check correctness and basic speed
- [ ] 1B vmemd + slab: apps link `libvmem` (per-process allocator over a backend: `vmem`, `uvm` or `cuda`) and call it directly; `vmemd` says device or host for each new chunk; memory in use never moves
  - [ ] 1Bi Design: API (`init`, `alloc`, `free`, `prefetch`, `evict`), chunk size, client-daemon protocol, device budget (include per-process context overhead)
  - [~] 1Bii Allocator, local mode (no daemon): SmallPool and LargePool, VRAM until `cuMemCreate` fails, then host; empty chunks released. Allocator and `cuda` backend written; `vmem` and `uvm` backends left
  - [ ] 1Biii `vmemd` broker: per-tenant usage, device budget, rule "device until full, then host; freed VRAM goes to the next request"; clients hold their own memory; client crash = socket close, daemon crash = clients fall back to local mode
  - [ ] 1Biv Hint-driven moves (next): `evict(p)` (data wanted later) and `free(p, Reuse::Later)` (block wanted later) mark a large chunk Cold or Cached, `prefetch(p)` marks it Hot; libvmem decides when to act: on VRAM pressure it destroys Cached chunks and moves Cold ones to host via backend `move` (same address), after `cuCtxSynchronize`; `alloc` reuses Cached chunks; small objects never move. Stubs in `libvmem.cuh`
  - [ ] 1Bv Test apps and multi-process runs; backends `vmem`, `uvm`, `cuda` for baselines; check correctness and basic speed
- [ ] 1C True compute sharing: run tenants under MPS and/or green contexts so kernels from different processes run at the same time (2A)
- [ ] 1D Smarter `vmemd` policies: quotas, fairness, which tenant gets VRAM, using hints and usage
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
- 2026-09-24: repo reorganised; PyTorch allocator study done on branch `pytorch-study` (preliminary); BoxD read.
- 2026-09-27: notes moved to `notes/`.
- 2026-09-28: Nixie and MSched found to cover most of the original pitch.
- 2026-09-29: direction set: Nixie-style multiplexing with true spatial sharing, pinned host fallback, general workloads, optional hints. Design phases replaced by prototype stages. PyTorch study ended.
- 2026-10-04: renamed `libvmem`; work moved to `main`.
- 2026-10-05: Nixie and Prism read (Nixie: basic design to expand; Prism: shows possible impact, too LLM-specific to build on). 2A/2B first pass: MPS needed for cross-process concurrency, one MPS client's fault kills all clients, green contexts inside MPS clients isolate SMs, VMM works under MPS; CUPTI launch-argument scan sees library kernels, hardware access counters unusable. 1A primitives written. 5A first run: device VMM calls nearly flat in size, host create about 89 us/MiB, back-to-back device calls stall about 2 ms. Report outline matched to the plan. Preliminary, 1 to 3 runs.
- 2026-10-06: 1Ai done: `primitives/runner.cu` checks a `.vm` action file in one pass, then runs it on one Manager with per-line logs (`--stable` for diffs, `-d` for state). `smoke.cu` moved to `tests/test1.cu` (+ `test1.vm`). `free(va)` no longer takes a size. 5A: VMM matches `cudaMalloc`; managed cost moves to first touch.
- 2026-10-07: primitives reorganised: `src/` (Manager, runner split into parts, shared fill/check), `actions/` (.vm), `tests/` (test1 basic, test2 full, test3 timing, `fuzz.py`), `bin/`. Runner: Manager output via `Options::out` (no fd capture), broken-context skip, `--pid`, header and PASS/FAIL. Fuzz: 50,000 lines in 600 random programs, 0 disagreements with the model.
- 2026-10-09: 5A done: per-call costs (device mostly fixed, set access largest; host linear in size), VMM matches `cudaMalloc`/`cudaFree`, managed cost moves to first touch (5 runs). Order of magnitude is enough: a daemon hides these calls. Per-call numbers are one run.
- 2026-10-14: `tests/test4.cu`: remapping a chunk while a kernel uses it crashes (illegal address) or silently loses writes. Moving memory in use safely needs the driver, so live remap, launch interception and `LD_PRELOAD` hooks are dropped. New core: vmemd + slab (1B). Old 1B (transparent hooks) and 1E (profile-guided hints) removed.
- 2026-10-19: `libvmem/` started: allocator over a backend interface (`vmem`, `uvm`, `cuda`; `cuda` written), SmallPool and LargePool as in PyTorch's caching allocator, all synchronization in the allocator. Moves are driven by hints that libvmem acts on under pressure, not by direct evict/prefetch calls (1Biv).
