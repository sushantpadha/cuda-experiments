# Tracker

**libvmem**: VMM-based Generalized GPU Memory Virtualization for Multi-Tenant applications. Several processes share one GPU at the same time. Apps call a slab allocator directly; a daemon (`vmemd`) picks VRAM or pinned host memory for each new chunk. Memory in use never moves.
Sushant Padha (S), Koduru Tejeswar (K). Mentor: Prof. Purushottam Kulkarni. IIT Bombay, CSE.

This file is the authoritative source for goals, subgoals, priorities and status. Other docs link here by ID.
Status: `[ ]` pending, `[~]` in progress, `[x]` done (remove and move to Log). Goals are numbered 1, 2; subgoals 1A, 1B; sub-subgoals 1Ai, 1Aii.
Timeline: full project about 1.5 months. No code deadline.
Current stage: 1B. The `vmem` backend, the `vmemd` broker and the client side are done and tested. Next: finish the multi-process comparison (1Biv), then MPS (1Biii).

## 1. Prototype (top priority)

Correctness first, then reasonable efficiency. 1A (v0 primitives) is done; see the Log.

- [~] 1B `libvmem` + `vmemd`: apps allocate through `libvmem`, `vmemd` places each new chunk on the device or the host, and memory in use never moves. The `cuda`, `uvm` and `vmem` backends and the `vmemd` broker work.
  - [ ] 1Biii MPS: run clients under MPS so their kernels run at the same time. `vmemd` may set MPS limits per client (2C)
  - [~] 1Biv Tests: multi-process runs with the `vmem`, `uvm` and `cuda` backends as baselines. Done for `vmem` with `vmemd`: `make test-pair` (two clients), `make test-workers-gemm` and `make test-workers-image` (four processes each). `ARGS="--backend vmem-local|cuda|uvm"` runs the same workers as baselines. `make trace-all` traces VRAM, DRAM and accepted hints over time on all four backends (`cuda`, `uvm`, `vmem` local, `vmem` + `vmemd`) and plots them in `libvmem/analysis/`; results in `libvmem/tests/RESULTS.md`. Left: several runs each, and image sizes that oversubscribe every time
  - [ ] 1Bv Non-blocking `evict`/`prefetch`. The host must run the `cuMemUnmap` + `cuMemMap` swap, and no CUDA event can order it
  - [ ] 1Bvi `vmemd` policies: quotas, fairness
  - [ ] 1Bvii Per-block content hashing: share identical blocks across processes
  - [ ] 1Bviii v2 additions: more locations than device and host (several NUMA nodes or GPUs; `Loc` and `used(Loc)` change); non-blocking `evict`/`prefetch` (1Bv); per-tenant priority in `vmemd`; `cudaMemAdvise` preferred location in the `uvm` backend if pages drift back; MPS support; optional `reserve_va`/`drop_va` calls on the `vmem` backend, so an app can reserve address space up front and map chunks into it

## 2. Look-ups

- [ ] 2C Can `vmemd` drive MPS limits? `nvidia-cuda-mps-control` offers `set_device_pinned_mem_limit <PID> <dev> <value>` and `set_active_thread_percentage <PID> <pct>`. Find out whether they apply to a running client, whether `<PID>` names the server or the client, and whether the cap makes a VMM `cuMemCreate` fail (the backend would then fall back to host, as on `ALLOC_FAIL`). MPS does not cap host memory

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

## 6. PyTorch (parked)

- [ ] 6A Caching allocator integration. The study is done (branch `pytorch-study`); nothing else is planned

## 7. Report (`report/`)

The outline predates the pivot to 1B, so it is stale. Rework it when report writing starts; in particular the design section still describes remap and launch hooks.

- [ ] 7A Introduction
- [ ] 7B PyTorch allocator section, from the existing study
- [ ] 7C Design section, from 1B
- [ ] 7D Experiments section. Needs 1Biv, 5B, 5E
- [ ] 7E Abstract and conclusion. Last
- [ ] 7F UVM vs. VMM section: drafted, needs 5B results
- [ ] 7G Related work section (Nixie, Prism, MSched, BoxD, GMLake, MPS and green contexts). Needs 4A
- [ ] 7H VMM call latency section (stub in place). Source: `experiments/vmm-latency/README.md`

## Log

- 2026-09-23: tracker and report skeleton created (project then named `vmadvise`).
- 2026-09-24: repo reorganised; PyTorch allocator study done on `pytorch-study`; BoxD read.
- 2026-09-27: notes moved to `notes/`.
- 2026-09-28: Nixie and MSched cover most of the original pitch.
- 2026-09-29: direction set: Nixie-style multiplexing with spatial sharing, host fallback, general workloads. PyTorch study ended.
- 2026-10-04: renamed `libvmem`; work moved to `main`.
- 2026-10-05: Nixie and Prism read. 5A first run.
- 2026-10-05: compute sharing (2A, `notes/compute-sharing.md`). Without MPS, processes take turns on the GPU; with MPS, their kernels run at the same time. VMM (device and host memory, remap, handle export) works under MPS, and MPS memory caps apply to VMM device memory only. Under MPS one client's bad access kills every client, so evicted memory must never stay unmapped. Host-backed reads run at about 13 GB/s against 187 GB/s on the device.
- 2026-10-05: eviction signals (2B, `notes/eviction-signals.md`). VMM ranges have no hardware access bits. A CUPTI launch-argument scan shows which buffers a kernel may touch, cheaply and including library kernels, but misses pointers stored in GPU memory. Kernel time grows as more data sits on the host, which gives a per-tenant pain signal. GPU block hashing runs at about 188 GB/s and finds unchanged blocks.
- 2026-10-07: v0 action runner, tests and fuzz done (0 disagreements in 600 programs).
- 2026-10-09: 5A done: VMM matches `cudaMalloc`; managed cost moves to first touch. A daemon hides call costs.
- 2026-10-14: 1A done: VMM-backed primitives (`primitives/`: Manager, runner, tests, fuzz). `primitives/tests/test4.cu` showed that remapping memory under a running kernel crashes or loses writes, so live remap is impossible in userspace. Dropped live remap, launch interception and `LD_PRELOAD`; pivoted to 1B.
- 2026-10-19: `libvmem/` started: backend interface, SmallPool and LargePool. `evict`/`prefetch` move a large chunk now; `free` is a plain free.
- 2026-10-20: `cuda` and `uvm` backends done; tests 1 and 2 pass on both.
- 2026-10-20: 8A done: PDFs other than `report/report.pdf`, tool state and `HANDOFF.md` purged from git history; `main` and `vmemd-slab` force-pushed (repo 44 MB to 5 MB).
- 2026-10-20: `vmem` backend (local mode) written; `test1v` and `test3 vmemd` pass. With `--resident`, `vmem` took 9.6 s against 14.2 s for `uvm` and 41.2 s for plain managed memory. `vmemd` skeleton and wire protocol written.
- 2026-10-20: `vmemd` broker side done: all messages, limits from a mandatory config file, file logging. `test4` checks it against a model with 3000 random calls. Messages without a reply are ordered only per tenant, so another tenant can briefly see old counts.
- 2026-10-20: 1Bi done: `vmem` backend in local mode; `test1v`, `test2v` and `test3 vmemd` pass without a daemon.
- 2026-10-20: 1Bii done: `vmem` is a `vmemd` client (location on create, grant on move, `AllocFail`, `MoveDone`, `Resync` on `reconnect()`; local mode without a broker or after losing it, 3 s receive timeout). Context overhead measured: 90 MiB bare context, 102 MiB with cuBLAS (lazy), 306 MiB eager, 81 MiB per MPS client plus 37 MiB for the MPS server; the broker adds `ctx_overhead_mib` on the first `Hello`. Tests reorganised: `test1`/`test2` on `c u v d`, `stress`, and `tests/daemon/` (`state`, `pair`, `workers`) that check the broker's `--debug` log against what clients saw. With `vmemd`, `stress vmemd --resident` took 11.4 s against 9.6 s in local mode.
- 2026-10-21: review of the client work. Config flag `auto_reconnect` (default off): `create` and `move` reconnect once while unlinked. `state_check.py` got `autoreconnect` and `config` scenarios (bad config files are refused). `worker.cu` takes `--backend vmem|cuda|uvm` and `--workload gemm|image`; the four-process test is split into `test-workers-gemm` and `test-workers-image`; first baseline runs in `libvmem/tests/RESULTS.md`.
- 2026-10-21: memory traces: `scripts/memtrace.py` records VRAM, DRAM and log-line counts while any command runs, `scripts/memplot.py` overlays the CSVs. One run each of both four-process tests on four backends. Image test: `vmem` local 3.3 s with 5.9 GB of DRAM, `vmem` + `vmemd` 2.3 s with DRAM capped at 3.4 GB and 18 of 72 hints refused, `uvm` 5.3 s, `cuda` out of memory.
