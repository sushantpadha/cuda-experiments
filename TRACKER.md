# Tracker

Project: **vmadvise**, a userspace residency-hinting library for GPU memory (VMM-based).
Authors: Sushant Padha (S), Koduru Tejeswar (K). Mentor: Prof. Purushottam Kulkarni. IIT Bombay, CSE.

**Current design phase:** 1 (use cases), not started.
Phases: 1 use cases, 2 requirements, 3 features, 4 API and consumer view, 5 design.
**Blocker:** the GPU needs a reboot (driver fatal error, 2026-09-24). Nothing that runs CUDA works until then.

Status: Done, In progress, Pending, Blocked. Owner `?` means unassigned.

## A. Report (`report/`)

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| A1 | Project name | S | Done | `vmadvise` |
| A2 | Template and build | S | Done | `make` builds `report/report.pdf` |
| A3 | UVM vs. VMM section | S | Done | Draft; needs experiment results (B2) |
| A4 | Introduction | ? | Pending | |
| A5 | PyTorch allocator section | ? | Pending | Needs C3 |
| A6 | Library design section | ? | Pending | Follows phases D1 to D5 |
| A7 | Experiments section | ? | Pending | Needs B2, E2 |
| A8 | Abstract, conclusion | ? | Pending | Last |

## B. UVM vs. VMM

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| B1 | Feature comparison | S | Done | In the report |
| B2 | Experiment: fault-driven paging vs. prefetch and remap | ? | Pending | Needs E1 and a working GPU |
| B3 | Further comparisons | ? | Pending | Order: thrash under oversubscription, first-touch cost, UVM with and without advise |

## C. PyTorch allocator study (branch `pytorch-study`)

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| C1 | Source study and notes | S | Done | `STUDY-GUIDE.md`, `NOTES.md` |
| C2 | Off/on `expandable_segments` experiments | S | Done | Preliminary, one session |
| C3 | Hands-on runs, snapshots, screenshots | ? | Blocked | GPU; use `allocator_lab.ipynb` |
| C4 | Move `testing/` scripts into notebook blocks | ? | Blocked | GPU: the conversion cannot be verified |
| C5 | Redo the `empty_cache` experiment with a second stream | ? | Pending | |

## D. Design (phases)

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| D1 | Phase 1: use cases and non-use cases | ? | Pending | |
| D2 | Phase 2: requirements | ? | Pending | After D1 |
| D3 | Phase 3: features | ? | Pending | After D2 |
| D4 | Phase 4: API and consumer view | ? | Pending | After D3 |
| D5 | Phase 5: design | ? | Pending | After D4 |
| D6 | Clarify open terms | ? | Pending | "shared memory optimization", "GPUBench" |

## E. Profiling and prototype

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| E1 | Learn Nsight Systems and Nsight Compute | ? | Pending | |
| E2 | Baseline traces of the existing experiments | ? | Pending | After E1 |
| E3 | Userspace slab allocator on VMM | ? | Pending | After D3 |

## F. Housekeeping

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| F1 | Untrack papers, slides, tool state, junk output | S | Done | 2026-09-24 |
| F2 | Purge them from git history and force-push `main` | S | Pending | Needs explicit permission to rewrite history |
| F3 | Reboot the GPU | S | Pending | See blocker |
| F4 | Ask the mentor about the anonymous manuscript in `references/` | S | Pending | |

## Log

- 2026-09-23: tracker and report skeleton created.
- 2026-09-24: repository reorganised into `experiments/`, `warmups/`, `references/`; `CLAUDE.md`, `README.md`, and this tracker rewritten.
