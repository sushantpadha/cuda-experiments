# CLAUDE.md

Guidance for AI coding agents working in this repository. Human readers: see `README.md`.

## Project

`vmadvise` is an R&D project on GPU memory management at the Department of Computer Science and Engineering, IIT Bombay. Authors: Sushant Padha, Koduru Tejeswar. Mentor: Prof. Purushottam Kulkarni.

Starting idea: a userspace library, built on CUDA's Virtual Memory Management (VMM) API, that gives applications `madvise`-style control over CPU-GPU memory residency. Goals so far:

- residency hints and priorities;
- memory sharing between processes, via per-block content hashing (share identical blocks, skip copying unchanged ones, copy-before-write);
- streaming-workload optimisation;
- userspace flexibility instead of driver-level changes;
- target workloads: naive kernels run in large batches (for example image processing) and Rodinia benchmarks;
- deeper study of UVM internals, and side-by-side comparison with Nixie, MSched and tuned UVM (advise plus prefetch).

**This is a vague starting point.** Goals, scope, and terms will change as the project develops. Treat them as provisional, not as fixed requirements.

Deliverables: a report (`report/`) and a prototype. **`TRACKER.md` is the single authoritative source for goals, subgoals, priorities and status.** Do not restate them elsewhere; link to tracker IDs instead. Current top priority is in the tracker.

**Goal-tracking rule (never confuse):** when the user says add, update or delete a goal or subgoal, find the matching higher-level goal (if it is new) or the exact goal/subgoal (if it exists) and make the change in `TRACKER.md` **and** `README.md` Status. That is the only meaning of those words. Tracker format: goals `1`, subgoals `1A`, sub-subgoals `1Ai` at most; keep it simple and human readable. When a subtask is done, mark it in the tracker (then move it to the Log). There is no progress file: the user records specifics in `notes/`. Keep the tracker always current.

## Notes and artifacts

- `notes/`: short, human-readable notes for the user to refer back to (for example `nixie.md`, `benchmarks-ideas.md`, `limitations.md`). Keep the set minimal. `notes/nixie.md` is the user's own handwriting: never edit it unasked.
- `notes/artifacts/`: your long-form research, compiled for you (`01-landscape-survey.md`, `remap-vs-uvm.md`, `RESOURCES.md`, `BENCHMARKS.md`). Pull from it when answering questions.
- When the user asks a new detailed question that needs research: ask first, then save the result as a dedicated file in `notes/artifacts/` for future reference.
- Style. Artifacts: always caveman full plus `plain-docs`; no slop. Pure notes (in `notes/`): write with caveman full plus `plain-docs`, using the Opus model (Agent with `model: opus`), short and human-sounding, like `notes/nixie.md`.

## Session start (mandatory)

1. Read `TRACKER.md`, and `HANDOFF.md` if it exists (local only, holds the design-phase plan).
2. Before any other work, ask the user which tracker items (by ID) and which phase they are working on today, using `AskUserQuestion`. Wait for the answer.
3. Stay inside those items. If something outside them comes up, mention it and ask. Do not act on it.

## Keeping the docs current

- Ask, occasionally and at natural stopping points, whether `TRACKER.md` needs updating.
- Whenever you ask about `TRACKER.md`, ask in the same question whether this `CLAUDE.md` needs updating too.
- Do not edit either file without the user's confirmation.
- Keep both lean. When updating, remove finished or obsolete entries instead of accumulating them; record dated facts in the tracker's Log.

## Design phases

The design work proceeds in gated phases. The current phase is recorded at the top of `TRACKER.md`.

1. Use cases (including workloads that do not benefit)
2. Requirements
3. Features
4. API, as seen by a consumer
5. Design

Know which phase you are in. Do not propose work that belongs to a later phase: no function signatures before Phase 3 is agreed, no architecture before Phase 4. The user drives the experiment, implementation, testing, and writing steps within a phase. Do not run that loop on your own initiative.

## Experiments

- Know the scope of the experiment you were asked to run and do not exceed it.
- State assumptions before running. Report results as measured, including negative results.
- Verify claims against the source, the documentation, or a run before asserting them.
- Check the GPU is healthy (see Hardware) before suspecting your code.

## Writing the report

- Write formal, simple academic prose. Follow the `plain-docs` skill: state things directly, no filler, no repeated hedging (collect caveats in one place), concrete numbers and names.
- **Before writing report text, ask the user what to refer to**: which sources, experiments, or sections it should draw on. Do not choose on your own.
- Cite only sources you have verified. Add them to `report/references.bib` and cite with `\citep{}`.
- Everything under `experiments/` and `warmups/` is scratch work that will be cleaned up or discarded. Never cite it as established fact. Attribute it as this project's preliminary observation.
- Build with `cd report && make`. The built `report/report.pdf` is committed.

## Domain primer

Facts come from the CUDA Programming Guide unless marked. **[Observed]** means seen in this project's experiments: a working assumption, not an established fact.

**Unified Memory (UVM)**

- `cudaMallocManaged` returns one pointer valid on host and device. The driver migrates pages on demand.
- Behaviour depends on the platform. This machine is the full-support, software-coherent case (page-fault based). Hardware-coherent platforms (for example Grace Hopper with NVLink-C2C) behave differently.
- Hints are advisory: `cudaMemAdvise` (preferred location, accessed-by, read-mostly) and `cudaMemPrefetchAsync` (stream-ordered asynchronous migration).
- Residency and eviction are decided by the driver for the whole device, not per process.

**Virtual Memory Management (VMM)**

- Driver API (`-lcuda`, explicit `cuInit` and context). Allocation is split into steps: `cuMemAddressReserve` (virtual range), `cuMemCreate` (physical handle, `CUmemGenericAllocationHandle`), `cuMemMap`, `cuMemSetAccess`. Teardown: `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`.
- Sizes must be multiples of the allocation granularity (`cuMemGetAllocationGranularity`).
- Backing memory is pinned, not migrated by page faults. It can live on the device or on host NUMA memory (`CU_MEM_LOCATION_TYPE_HOST_NUMA`).
- A handle can be exported (`cuMemExportToShareableHandle`, POSIX file descriptor) and imported by another process.
- **[Observed]** A handle cannot be mapped partially: `cuMemMap` needs offset 0 and the full handle size.
- **[Observed]** No fault handler covers these ranges. A kernel that touches an unmapped address crashes.
- **[Observed]** Cross-process writes have no automatic ordering. Synchronise before handing off.

**Related**

- PyTorch's caching allocator with `expandable_segments` is built on VMM. Details: `experiments/pytorch-vmm-study/` (`STUDY-GUIDE.md`, on branch `pytorch-study`).
- `cudaMallocAsync` memory pools, MPS, and MIG are the usual points of comparison.

## Hardware and toolchain

- RTX 4050 Laptop (compute capability 8.9, 20 SMs, about 5.7 GB, 2 copy engines), CUDA 13.0 (`nvcc` 13.0.88), driver 580. Host NUMA-backed VMM works.
- `nvidia-smi` can look healthy while CUDA is broken. If CUDA reports error 999 or "unknown error", run `journalctl -k | grep -i Xid`. A driver fatal error needs a reboot.

## Layout

```
report/        LaTeX report (TMLR-style); make -> report/report.pdf
TRACKER.md     authoritative goals, subgoals, status, current phase
notes/         short human notes; notes/artifacts/ holds long-form research for Claude
experiments/   scratch experiments: VMMVector (growable vector on VMM),
               VMMRemapShared (remap primitive + multi-process allocator),
               pytorch-vmm-study (allocator study notes)
warmups/       small standalone kernels and device probes
references/    README.md with links; local-only PDFs are not tracked
```

Branches: `main` is the only published branch. `pytorch-study` and `vmm-experiments` hold further experiments and stay local unless the user says otherwise.

## Build and run

No top-level build. Each directory is standalone:

```
cd experiments/VMMVector && make      # or: make debug   (adds -g -G -DDEBUG)
./main 1000000 4 6
```

Others: `nvcc file.cu -o out` (add `-lcuda` for driver-API code).

## Conventions

- `common.cuh`: `CUDA_CHECK` for runtime calls, `CU_CHECK` for driver calls, `DPRINT` under `-DDEBUG`.
- C++17. The driver API needs an explicit `cuInit` and primary context (see `init_driver_state` in `VMMVector/main.cu`).
- `.gitignore` is allowlist-style: add extensions explicitly to track them. `*.txt` files are captured run output, not source.

## Git and files

- Commit only when asked. Never force-push or rewrite history unless the user explicitly asks.
- Other sessions may be working in the repository. Check `git status` and re-read a file before editing it, and never revert changes you did not make.
- Do not track PDFs other than `report/report.pdf`, nor tool state (`.claude/`, `.serena/`, `.vscode/`), nor `HANDOFF.md`.
- Keep the `Co-Authored-By` trailer on commits. Omit `Claude-Session` links: they are private.
