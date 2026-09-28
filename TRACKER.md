# Tracker

**vmadvise**: userspace residency-hinting library for GPU memory, built on VMM.
Sushant Padha (S), Koduru Tejeswar (K). Mentor: Prof. Purushottam Kulkarni. IIT Bombay, CSE.

This file is the authoritative source for goals, subgoals, priorities and status. Other docs link here by ID.
Status: `[ ]` pending, `[~]` in progress, `[x]` done (remove and move to Log). Goals are numbered 1, 2; subgoals 1A, 1B; sub-subgoals 1Ai, 1Aii.
Timeline: about 3 to 4 days for basic ideation of all design phases, then mentor discussion and redo. Full project about 1.5 months. No code deadline.
Design phase now: 1 (use cases).

## 1. Read related work (top priority)

- [~] 1A Nixie paper: understand how it works. `references/nixie-paper.pdf`, notes in `notes/nixie.md`
- [ ] 1B GMLake paper: understand how it works. `references/gmlake-paper.pdf`
- [ ] 1C vAttention and vTensor beyond the abstract
- [ ] 1D Ask Soham for verified page-fault cost numbers and papers
- [ ] 1E Verify or drop the remaining unverified related-work entries
- [ ] 1F Deeper contrast: read MSched, Prism, Concordia in full and kvcached code; update `notes/artifacts/research-directions.md`

## 2. Design, in gated phases

- [~] 2A Phase 1: use cases and non-use cases
  - [~] 2Ai Anchor cases: A (batch streaming, naive kernels), C (multi-process residency), D (grow in place). Later: F (cross-process sharing), B (Rodinia-style), E (oversubscribed training or inference). Revisit against the survey.
  - [ ] 2Aii Non-use cases
  - [~] 2Aiii Positioning: BoxD adds control to the UVM driver; Nixie is transparent whole-app multiplexing on VMM. What does a hint-carrying interface add over Nixie? Also MSched (kernel-arg working sets, modified driver). Directions D1 to D4 in `notes/artifacts/research-directions.md`
  - [ ] 2Aiv Write `01-usecases.md`: anchors, non-use cases, claimed gap in one sentence
- [ ] 2B Phase 2: requirements. Includes single vs multi-process and consumer form (explicit API first)
- [ ] 2C Phase 3: features
- [ ] 2D Phase 4: API and consumer view
- [ ] 2E Phase 5: design
- [ ] 2F Candidate feature ideas, to test in Phase 3 (`research-directions.md` D1, D2)
  - 2Fi Per-block content hashing: skip copy-out of unchanged blocks on eviction; share identical blocks across processes (truer shared GPU memory); copy-before-write for shared blocks (real COW impossible, no write fault)
  - 2Fii `madvise`-style hints (read-only, will-need, don't-need) feeding the same machinery; read-only makes sharing safe

## 3. UVM vs. VMM experiments

- [ ] 3A Fault-driven paging vs. prefetch and remap. Baselines: plain UVM and UVM with advise and prefetch, always both. Needs 4A
- [ ] 3B Further comparisons: thrash under oversubscription, first-touch cost, UVM with and without advise
- [ ] 3C Read `uvm_pmm_gpu.c` and `uvm_gpu_access_counters.c` (driver 580) to settle the hotness claims
- [ ] 3D Side-by-side runs on this machine, same workloads: Nixie, plain UVM, UVM with advise and prefetch, nvshare, MSched if released, kvcached, manual `cudaMalloc`. All related-work contrast so far is from papers only

## 4. Profiling and prototype

- [ ] 4A Learn Nsight Systems and Nsight Compute
- [ ] 4B Baseline traces of the existing experiments. After 4A
- [ ] 4C Userspace slab allocator on VMM. After 2C

## 5. PyTorch allocator study (branch `pytorch-study`)

- [ ] 5A Hands-on runs, snapshots, screenshots. `allocator_lab.ipynb`, checklist in the folder README
- [ ] 5B Redo the `empty_cache` experiment with a second stream. `STUDY-GUIDE.md` section 10

## 6. Report (`report/`)

- [ ] 6A Introduction
- [ ] 6B PyTorch allocator section. Needs 5A
- [ ] 6C Library design section. Follows 2A to 2E
- [ ] 6D Experiments section. Needs 3A, 4B
- [ ] 6E Abstract and conclusion. Last
- [ ] 6F UVM vs. VMM section: drafted, needs 3A results

## 7. Housekeeping

- [ ] 7A Purge tracked PDFs and tool state from git history and force-push `main` (S). Needs explicit permission to rewrite history
- [ ] 7B Ask the mentor about the anonymous BoxD manuscript in `references/`: source and how to cite (S)

## Log

- 2026-09-23: tracker and report skeleton created; project name `vmadvise`; report build and template done.
- 2026-09-24: repo reorganised into `experiments/`, `warmups/`, `references/`; papers, slides and tool state untracked; docs rewritten; GPU rebooted; PyTorch study consolidated on branch `pytorch-study`, source study and off/on experiments done (preliminary, `experiments/pytorch-vmm-study/STUDY-GUIDE.md`). Decided: "shared memory optimization" means cross-process sharing (case F, parked); "GPUBench" dropped, Rodinia only; BoxD read in full.
- 2026-09-27: added top priority 1 (Nixie, GMLake); notes moved to `notes/`, long-form research to `notes/artifacts/`; `PROGRESS.md` removed, tracker simplified and renumbered.
- 2026-09-28: Nixie and MSched cover most of the original pitch; new directions (hashing, sharing, copy-before-write, measurement) in `notes/artifacts/research-directions.md`. Added 1F, 2F, 3D.
