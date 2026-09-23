# Tracker — GPU Memory Management Project

Authors: Sushant Padha (24B1057), Koduru Tejeswar (24B0918)
Mentor: Prof. Purushottam Kulkarni
Dept. of CSE, IIT Bombay

Report source: `report/`. Code: dirs listed in `CLAUDE.md`.

## Threads

### 1. UVM vs VMM study
- [ ] Feature matrix: UVM (`cudaMallocManaged`) vs VMM (`cuMem*`) — what each covers, where UVM lacks (pinned control, explicit residency, multi-proc sharing)
- [ ] Write up in `report/sections/03_uvm_vs_vmm.tex`

### 2. PyTorch caching allocator study
- [x] Source/doc read on `CUDACachingAllocator` + `ExpandableSegment` (`pytorch-vmm-study/`)
- [x] Runnable stress tests, expandable_segments on/off (`pytorch-vmm-study/testing/`, `RESULTS.md`)
- [ ] Visualizer experiments (memory snapshot tool) — capture + annotate 2-3 traces
- [ ] Write up in `report/sections/04_pytorch_allocator.tex`

### 3. Library design
- [ ] Desired feature list: madvise-like residency hints/priorities, shared-mem optimization, streaming workload hints
- [ ] Use-case list: naive batch kernels (img processing), Rodinia/GPUBench-style workloads
- [ ] API sketch (function-level, not impl)
- [ ] Design doc, backed by `VMMVector`/`VMMRemapShared`/`VMMSlab`/`VMMExplore` findings
- [ ] Write up in `report/sections/05_library_design.tex`

### 4. Experimentation
- [ ] Nsight Systems: profile existing VMM* code, baseline traces
- [ ] Nsight Compute: kernel-level counters where relevant
- [ ] Capture traces for report figures
- [ ] Write up in `report/sections/06_experiments.tex`

### 5. Report assembly
- [ ] Project name
- [ ] Abstract (last)
- [ ] Intro
- [ ] Conclusion
- [ ] Pass over full PDF, figures, refs

## Log

- 2026-09-23: tracker + report skeleton created.
