# Benchmarks and comparison targets

Test targets for `vmadvise`. Nothing run. All TBD until chosen (tracker 2A to 2C), scheduled (3A, 4B). Sources: `HANDOFF.md` section 7, `notes/artifacts/remap-vs-uvm.md`, `notes/artifacts/01-landscape-survey.md`.

Status values: TBD (discussed, not scheduled), Chosen, Set up, Run.

## Workloads

| Workload | Use case | Status | Notes |
|---|---|---|---|
| Batch image/video, naive kernels, data > VRAM | A | TBD | Streaming tiles, sequential. Anchor. |
| Multi-process jobs sharing one GPU | C | TBD | Per-process residency, priorities. Anchor. Compare BoxD 3-process setup. |
| Growing buffers (append, resize, KV-cache-like) | D | TBD | Anchor. Grow in place vs. copy on realloc. |
| Cross-process buffer handoff | F | TBD | Parked. |
| Rodinia, regular kernels (e.g. hotspot, SRAD) | B | TBD | Rodinia only; "GPUBench" dropped. Working set may exceed VRAM. |
| Rodinia, irregular kernels (e.g. BFS) | B | TBD | UVM thrash expected on irregular access. |
| Oversubscribed training/inference, host as 2nd tier | E | TBD | Schedule-driven prefetch. Later. |
| UVM vs. VMM microbench: fault paging vs. explicit prefetch/remap, oversubscribed | all | TBD | Tracker 3A, needs 4A. Report section 2.4. |
| Non-beneficiaries (fits in VRAM, single-tenant static alloc) | none | TBD | Control runs, overhead; see `HANDOFF.md` section 6. |

## Baselines and competitors

| System | Layer | Status | Where to get it | Notes |
|---|---|---|---|---|
| `cudaMalloc` with manual copies | Application | TBD | CUDA toolkit | Hand-tuned double buffering for case A. |
| UVM, plain (`cudaMallocManaged` only) | Driver | TBD | CUDA toolkit | Nixie's "UVM" baseline hooks only `cudaMalloc`, `cudaFree`, `cudaMemGetInfo`; its numbers use this. |
| UVM with `cudaMemAdvise` and `cudaMemPrefetchAsync` | Driver | TBD | CUDA toolkit | Fair baseline. Both UVM baselines needed (`remap-vs-uvm.md`). |
| `cudaMallocAsync` memory pools | Runtime | TBD | CUDA toolkit | |
| PyTorch caching allocator, with and without `expandable_segments` | Framework | TBD | Branch `pytorch-study` | |
| **Nixie** (Xu, Wang, Ren, Chen, Zhuo; Duke University) | Userspace shim plus daemon on VMM | TBD | Paper: <https://arxiv.org/abs/2601.11743>. Code: <https://github.com/XOR-op/Nixie> (Rust; read at eebf583). OSDI '26 per README. | Runnable: `nixie daemon`, `nixie run <cmd>`. Hardware differs (RTX 5090 vs. RTX 4050 here). Details: `nixie-summary/`. |
| **nvshare** (Alexopoulos and Mitropoulos; ICSE-Companion '24, pp. 16-20) | UVM-based time slicing of shared GPU | TBD | Paper: Nixie ref. [3]. Repo <https://github.com/grgalex/nvshare> (confirmed via `gh`, 2026-09-27). | Nixie uses windows W=4 s, W=30 s (30 s default). |
| BoxD | Kernel driver, cgroup limits on UVM | TBD | `references/boxd_socc_winter2026_v3 1.pdf` | Anonymous; no known code. Case C only. |
| GMLake | VMM stitching in PyTorch allocator | TBD | <https://github.com/antgroup/glake> | Case D (fragmentation). |
| TGS | UVM-based priority sharing | TBD | Nixie ref. [34] | Two apps only, explicit priorities; Nixie excluded it from microbenchmarks. |

## Metrics

Wall time, stall per kernel, bytes migrated, resident set per tenant over time, allocator overhead, CPU pinned memory. Time to first token, context-switch time if Nixie-style workload. Tools: Nsight Systems (UVM fault, migration events), Nsight Compute (tracker 4A).

## Open items

- Pick anchor workloads (tracker 2Ai). Keep table in step.
- Check nvshare and Nixie licences before running them.
- One RTX 4050 (about 5.7 GB). Nixie, nvshare used 24 to 32 GB cards, models nearly filling VRAM; scale down to oversubscribe.
- Nixie workloads: whole apps (llama.cpp, SGLang, ComfyUI). Match, or synthetic kernels?
