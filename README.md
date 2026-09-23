# CUDA Experiments — vmadvise R&D Project

R&D project on GPU memory management: a userspace residency-hinting library
(`vmadvise`) built on CUDA's Virtual Memory Management (VMM) API.

Authors: Sushant Padha (24B1057), Koduru Tejeswar (24B0918)
Mentor: Prof. Purushottam Kulkarni
Department of Computer Science and Engineering, IIT Bombay

Report source: `report/` (`cd report && make` builds `main.pdf`). Task
tracker: `TRACKER.md`.

## Status

- **Report** — `report/`. Section stubs in place, `UVM vs.\ VMM` section
  written.
- **PyTorch caching-allocator experiments** — to be done. Notes so far in
  `pytorch-vmm-study/`.
- **Basic VMM API demo** — done. `VMMVector/`: growable GPU vector on
  `cuMemCreate`/`cuMemMap`/`cuMemSetAccess`.
- **UVM vs.\ VMM experiments** — pending. Measuring the actual gap between
  UVM's fault-driven migration and VMM's explicit remap (`VMMRemapShared/`)
  on this hardware.
- **Feature list / requirements / workloads** — pending. Need the target set
  of features, and which workloads benefit from a `vmadvise`-style library
  versus which don't.
- **Nsight (tracing/profiling)** — pending. Needed before the experiments
  above produce usable numbers.
- **Userspace slab allocator on VMM API** — pending. Blocked on the feature
  list above being settled first.

## Layout

See `CLAUDE.md` for the full directory breakdown, hardware, build/run
instructions, and code conventions.
