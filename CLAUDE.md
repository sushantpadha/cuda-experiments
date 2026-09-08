# CLAUDE.md

R&D scratch repo for CUDA experiments. Goal: find something concrete to extend
(library / driver extension / runtime tweak) in GPU memory management, async GPU
I/O, scheduling, or CUDA-vs-AMD primitives.

Current thread: a VMM-based allocator that can remap backing store to host DRAM
and be shared across processes for multi-tenant GPU use. Progress tracker:
https://claude.ai/code/artifact/dcf94637-253f-4233-9124-d96bfa7c5c09

## Hardware / toolchain

- GPU: RTX 4050 Laptop, compute capability 8.9, 20 SMs, ~5.7 GB, 2 copy engines.
- Full unified memory support with software coherency; host NUMA-backed VMM works.
- CUDA 13.0 (`nvcc` V13.0.88). Driver API needs `-lcuda`.

## Layout

- `VMMVector/` — base work. Growable GPU vector built on the low-level VMM driver
  API (`cuMemCreate` / `cuMemMap` / `cuMemSetAccess`), chunked reserve+map,
  chunk retention on shrink. `main.cu` is a staged test harness
  (`./main N chunk_mb count`). Notes in `VMMVector/README.md`.
- `VMMRemapShared/` — the extension. `cudaremap` primitive (`remap.cuh`: swap a
  VA range's backing device<->HOST_NUMA in place) + a multi-tenant allocator
  (`owner.cu` / `subscriber.cu`) that shares one VMM segment across processes and
  remaps chunks live via a revoke/remap handshake. `./run.sh` → `example_output.txt`.
- `pytorch-vmm-study/` — notes on VMM usage in PyTorch's `CUDACachingAllocator`
  (`ExpandableSegment`), mapped against `VMMVector`, with source/doc/blog links.
- `transpose/`, `vecAdd/` (`manual.cu` = explicit copies, `uvm.cu` = unified
  memory), `vecadd.cu` — small standalone kernels.
- `props.cu` / `test.cu` — device query + sanity probes.
- `cuda-programming-guide.pdf` — reference.

## Build / run

No top-level build. Each dir is standalone:

```
cd VMMVector && make        # or: make debug   (adds -g -G -DDEBUG)
./main 1000000 4 6
```

Others: `nvcc file.cu -o out` (add `-lcuda` for driver-API code).

## Conventions

- `common.cuh`: `CUDA_CHECK` for runtime calls, `CU_CHECK` for driver calls,
  `DPRINT` under `-DDEBUG`.
- C++17. Driver API requires explicit `cuInit` + primary context (see
  `init_driver_state` in `main.cu`).
- `.gitignore` is allowlist-style: add extensions explicitly to track them.
- `*.txt` files are captured run output, not source.
