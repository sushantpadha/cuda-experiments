# Tracker

**libvmem**: VMM-based Generalized GPU Memory Virtualization for Multi-Tenant applications. Several processes share one GPU at the same time. Apps call a slab allocator directly; a daemon (`vmemd`) picks VRAM or pinned host memory for each new chunk. Memory in use never moves.
Sushant Padha (S), Koduru Tejeswar (K). Mentor: Prof. Purushottam Kulkarni. IIT Bombay, CSE.

This file is the authoritative source for goals, subgoals, priorities and status. Other docs link here by ID.
Status: `[ ]` pending, `[~]` in progress, `[x]` done (remove and move to Log). Goals are numbered 1, 2; subgoals 1A, 1B; sub-subgoals 1Ai, 1Aii.
Timeline: full project about 1.5 months. No code deadline.
Current stage (from 2026-11-16, about one week): a basic PyTorch integration (6B, 6C), the report (goal 7) with its slides (7I), and the 1Biv repeat runs the results need. Everything else is parked under "After the report".

## 1. Prototype

1A (v0 primitives) is done; see the Log. 1B (`libvmem` + `vmemd`) has the `cuda`, `uvm` and `vmem` backends and the `vmemd` broker working, and is on hold apart from 1Biv.

- [~] 1Biv Tests: multi-process runs with the `vmem`, `uvm` and `cuda` backends as baselines. `make test-pair`, `make stress`, `make test-workers-gemm`, `-image` and `-mix` pass; `make trace-all` traces all five backends into `libvmem/analysis/`, results in `libvmem/tests/RESULTS.md`. Left: run each four-process test 5 times and report median and range (needed for 7E)

## 6. PyTorch integration (torch-vmem)

Plan, design choices and hypotheses H1 to H3: `notes/pytorch-plan.md`. Basic version only; the trace study (6A) is not a gate.

- [ ] 6B PyTorch on `libvmem`, no hints: C ABI `.so` (`alloc`, `free`, `evict`, `prefetch`, `where`), LargePool chunk cache, one mutex, loaded with `CUDAPluggableAllocator`. Run one script on `cuda`, `uvm`, `vmem` and `vmem` + `vmemd`: overhead when it fits, spill or out of memory when it does not
- [ ] 6C Simple blocking hints: a `TorchDispatchMode` hook evicts tensors after their last use and prefetches a few ops ahead, from the previous iteration's order, on one workload larger than VRAM. Compare with the same hints on `uvm`

## 7. Report (`report/`, top priority)

`report/report.pdf` builds; each section lists its gaps as red TODOs (writing) and red REFs (sources). Writing rules: `report/report-rules.md`.

- [ ] 7A Introduction: problem, goal, path taken and contributions; a source on shared-GPU usage
- [ ] 7B Background: `cudaMalloc` paragraph; table of which calls block, with the VMM latency plots (data from 5A); UVM vs. VMM semantics table from the docs and our observations; whether `expandable_segments` is on by default; two sentences on MPS; how an OS swaps pages
- [ ] 7C Survey: Nixie, Prism, gap table. "Other systems" is dropped unless MSched (4A) gets read
- [ ] 7D First design: transparent tiered-memory idea, first VMM allocators, v0 primitives and the remap flaw (with a timeline figure), why not modify the driver
- [ ] 7E `libvmem`: API, `vmemd` (with an architecture figure), results (needs 1Biv), where the gain comes from
- [ ] 7F PyTorch: why PyTorch, the extension points, how `libvmem` plugs in (with a figure), 6B and 6C results; the rest of the plan as future work
- [ ] 7G Limitations, conclusion and future work, then the abstract last
- [ ] 7H References: verify every REF and add it to `report/references.bib`
- [ ] 7I Slides: Beamer talk in `report/`, built from the finished report

## After the report

Parked; IDs kept so other docs still resolve.

- 1Biii MPS for `libvmem` clients; 1Bv non-blocking `evict`/`prefetch`; 1Bvi `vmemd` policies (quotas, fairness); 1Bvii per-block content hashing; 1Bviii v2 additions (more locations, priorities, `reserve_va`)
- 2C whether `vmemd` can drive MPS limits on a running client
- 3A LLM inference hints (KV cache vs. weights); 3B streaming workloads larger than VRAM
- 4A MSched; 4B GMLake, vAttention, vTensor, Concordia; 4C page-fault cost numbers from Soham; 4D verify the remaining related-work entries; 4E tensor offload in DNN frameworks (DeepUM, vDNN, SwapAdvisor, Capuchin, ZeRO, FlexGen)
- 5B, 5C UVM vs. VMM experiments (zero-copy break-even, move cost, overlap, `cuStreamWaitValue32` gate, thrash, host memory); 5D read the UVM driver; 5E side-by-side runs with Nixie and others; 5F Nsight Systems and Compute. Runs listed in `notes/pytorch-plan.md`
- 6A op trace study; 6D async moves; 6E multi-tenant reclaim through `vmemd`. Plan in `notes/pytorch-plan.md`

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
- 2026-10-21: tests regrouped into unit, stress and workload. `test1.cu`/`test2.cu` take the backend letter (`make test-unit` runs all eight) and check more pool edges; test 2's out-of-memory step stops at VRAM + 1 GiB. `stress.cu` replaced by `make stress`: four processes of 2000 random calls each, data and counts checked after every call, against the broker's log. Image workload now oversubscribes every run (6144 MiB) and goes over its images twice; new `raw` baseline (managed memory, no hints) and mix test (gemm and image tenants together). First runs, image test: `vmem` + `vmemd` 3.5 s, `vmem` local 4.5 s, `uvm` 9.0 s, `raw` 25.2 s, `cuda` out of memory.
- 2026-11-03: plan changed: UVM vs. VMM experiments (5B, 5C, now split into concrete runs), then PyTorch (goal 6, unparked). 1B on hold. PyTorch source read at `f904ab8` (2.16.0a0): `CUDAPluggableAllocator` sends every CUDA storage through two C functions; `MemPool` with `no_split` keeps the caching allocator but routes one thread only; `TorchDispatchMode` and the CUDA sanitizer give per-op read and write sets; `save_on_cpu` copies without lookahead. Plan in `notes/pytorch-plan.md`.
- 2026-11-04: report restructured into eight sections (introduction, background, survey, first design, `libvmem`, PyTorch plan, limitations, conclusion), with red TODOs and REFs for every gap; BoxD dropped from it. Tracker cut to the report and 1Biv; other items parked.
- 2026-11-16: final plan: a basic PyTorch integration (6B, 6C unparked) plus the report and Beamer slides (7I), in about one week. 6A, 6D, 6E, 5B and 5C stay parked as future work.
