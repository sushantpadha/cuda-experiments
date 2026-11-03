# Plan: UVM vs. VMM experiments, then PyTorch

> AI-generated :) plan, 2026-11-03. Tracker 5B, 5C, 6. PyTorch read at `f904ab8` (2.16.0a0), not run. Nothing here measured yet

## Why PyTorch

- CUDA never says which memory a kernel touches. so far: hints explicit and blocking, or move the whole app while it is idle (Nixie)
- a PyTorch op does say: every aten op names its tensors, and its schema marks the ones it writes (`Tensor(a!)`). known on the host before any kernel is queued
- so the dispatcher is the fault handler VMM never had: before an op, make its inputs resident (or accept zero-copy); after it, record an event = last use
- host-backed VMM chunk stays mapped, so a wrong guess is slow (zero-copy, ~13 GB/s observed), not a crash. the policy can be approximate
- tensor keeps its address through a move: no `.to()`, no new tensor, autograd, optimizer and CUDA graph pointers stay valid
- multi-tenant: the tenant knows which chunks have no queued kernels, so `vmemd` can take VRAM back from a *running* tenant, per tensor, not the whole app

## What PyTorch offers (read in source)

- `CUDAPluggableAllocator(so, alloc, free)` + `change_current_allocator`: every CUDA storage goes through two C functions, `void *alloc(size, device, stream)` and `void free(ptr, size, device, stream)`. set before the first CUDA allocation. cuBLAS workspace goes through it too. one storage = one call
  - lost: caching, expandable segments, stats (`getDeviceStats` throws, so `memory_allocated()` breaks)
  - `free` comes on the host when the tensor dies, kernels may still use it. stream passed = allocation stream
- `MemPool(allocator, no_split=True)` + `use_mem_pool(pool)`: caching allocator keeps caching, takes segments from us, one tensor per segment. routes the current thread only (backward runs on autograd's device thread, not routed). everything else stays `cudaMalloc`, invisible to `vmemd`
- `TorchDispatchMode` (`torch/utils/_python_dispatch.py`): every aten op with args and outputs, eager mode. `torch/cuda/_sanitizer.py` (CSAN) already turns the schema into per-op read and write sets (`ArgumentHandler`): copy it
- `saved_tensors_hooks(pack, unpack)`: tensors saved for backward. built-in `save_on_cpu` copies to a new pinned tensor at pack and `.to(cuda)` at unpack, no lookahead
- module forward/backward pre/post hooks: layer granularity, cheapest
- C++ `RecordFunction` global callback with inputs and outputs (`aten/src/ATen/record_function.h`), if Python per-op cost is too high
- blind spots: `torch.compile`/Triton kernels skip the dispatcher; CUDA graph replay; extensions with raw pointers; NCCL. their chunks never move

## Choices

- global pluggable allocator, not `MemPool`: one storage = one chunk; `vmemd` sees every byte; PyTorch cannot reuse an evicted block behind our back
- no PyTorch fork, no rebuild. public hooks only. PyTorch source is for reading
- `libvmem` changes, small:
  - C ABI `.so`: `alloc`, `free`, and `evict`/`prefetch`/`where` by pointer
  - LargePool chunk cache keyed by size. today every alloc is a `cuMemCreate` (0.2 to 0.6 ms) and every large free a `cuCtxSynchronize`. trim on `NoMemory`, then retry
  - one mutex (the autograd thread allocates too)
  - reuse only on the same stream, PyTorch's own rule. prototype = one compute stream
- moves blocking first (as now), async later (6D)

## UVM vs. VMM experiments (5B, 5C)

one small `.cu` each in `experiments/uvm-vs-vmm/`, 5 runs, median and range. each answers a design question

| ID | Experiment | Question it answers |
|---|---|---|
| 5Bi | kernel reads a buffer k times (k = 1..8, 64 MiB..1 GiB): VMM host chunk (zero-copy), UVM preferred CPU + accessed-by GPU, UVM faults, UVM prefetch, VMM move then read. plus cuBLAS GEMM with A on host, sweep M | when is "leave it on host" fine? break-even k. can decode run on host weights? |
| 5Bii | move cost per size, both ways. VMM split into create, copy, unmap, map, access, release; UVM `cudaMemPrefetchAsync` to GPU and to CPU. host time vs. GPU time. both directions at once | what a hint costs, how much of it blocks the host |
| 5Biii | same moves while an unrelated kernel runs | can moves hide behind compute; kernel slowdown; the ~2 ms set-access stalls |
| 5Biv | `cuStreamWaitValue32` gate: device attribute, host write to kernel start latency, kernel queued behind the gate while its chunk is swapped (fuzz, like `primitives` test 4) | can an async remap be ordered at all (1Bv, 6D) |
| 5Ci | working set 0.5x..2x VRAM: sequential, repeated, random 4 KiB, sparse. UVM plain, UVM advise + prefetch, VMM device first then host, no moves. nsys for bytes migrated | where UVM wins (sparse, unpredictable) and loses (thrash) |
| 5Cii | UVM host memory kept behind GPU pages (CPU-written, prefetched, `MemAvailable`); does prefetch to CPU free VRAM at once (`cudaMemGetInfo`); first-touch cost | host memory accounting, `vmemd` vs. UVM |

output: one "UVM vs. VMM semantics" table (who decides, unit, faults, ordering, move cost, host memory, multi-process, MPS, graphs) for 7F

## PyTorch steps (6)

one change per step, measured against the baselines on the same script before the next

**6A trace study, no allocator change.** gate: op level worth it?
- dispatch mode logs per op: name, storages read and written (pointer, bytes). kernel times from a separate `torch.profiler` run
- 3 workloads that fit the 6 GB laptop, traced on the GPU: GPT-2-small-sized training (`nn.TransformerEncoderLayer` stack, no download), ResNet-50 training, decode on a synthetic fp16 transformer (fewer layers for the trace; pattern repeats per layer)
- out: footprint over time; share of bytes in storages >= 2 MiB (movable); per-storage idle gap vs. move time (2 x size / 13 GB/s); same op order every iteration?; hook cost per op
- stop if most bytes are small or gaps shorter than moves: then module level only

**6B PyTorch on `libvmem`, no hints.** gate: overhead small when it fits (my guess at a bar: 5%)
- shim, cache, mutex. backends `cuda`, `uvm`, `vmem`, `vmem` + `vmemd`, same PyTorch script
- fits: time, peak VRAM, waste from 2 MiB rounding vs. native caching allocator
- doesn't fit: native OOMs, `vmem` spills to host (zero-copy), `uvm` faults. first real result

**6C semantic hints, blocking.** gate: does VMM beat UVM given the same hints?
- policy from the 6A trace (iterations repeat): after an op, evict storages whose next use is far; at op i, prefetch storages op i+d needs. Belady over last iteration. activations keyed by (op index, output index)
- wait on the chunk's last-use event, not `cuCtxSynchronize`: add no-sync `evict`/`prefetch` to the C ABI
- training activations > VRAM: vs. `save_on_cpu`, vs. activation checkpointing
- inference weights > VRAM: vs. hand-written `.to()` layer-window prefetch on a side stream
- same hooks on `uvm` with `cudaMemPrefetchAsync` on the compute stream (non-blocking, fair). the key baseline
- if UVM + hints matches VMM, single-tenant speed is not the contribution: say so, go to 6E

**6D async moves (with 1Bv).** needs 5Biv
- evict: background thread waits the last-use event, copies on a copy stream, then swaps. an op on that tensor meanwhile cancels it (write) or waits (swap window)
- prefetch: queue ops behind a `cuStreamWaitValue32` flag; host thread sets it after the swap. no gate support: block only dispatch of ops on that tensor
- until the swap, kernels read the host copy (slow, correct). swap only with no queued user

**6E multi-tenant reclaim through `vmemd`**
- new message, broker to tenant: `Reclaim(bytes)`. tenant answers at its next op (dispatch hook = safe point), evicts idle chunks (last use done, next use far)
- breaks broker rule K4 (never makes a tenant evict): rewrite it
- scenario: inference tenant idle with bursts + training tenant that wants more VRAM, two processes; MPS later (1Biii)
- vs. UVM for both, `cudaMalloc` (training sized down or OOM), whole-app swap on idle (Nixie-like: evict everything)

## Metrics

iteration time or tokens/s; peak VRAM, peak pinned DRAM, swap; bytes moved each way; host time blocked in moves; hook cost per op; GPU idle time (nsys). 5 runs, median and range

## Code budget (guess)

| Part | Where | Lines |
|---|---|---|
| 5B, 5C experiments | `experiments/uvm-vs-vmm/` | ~600 CUDA |
| tracer + analysis | `libvmem/torch/` | ~250 Python |
| C shim, cache, mutex | `libvmem/torch/`, `src/libvmem.cuh` | ~150 C++ |
| policy hooks | `libvmem/torch/` | ~250 Python |
| `Reclaim` | `daemon/`, `src/backends/vmem.h` | ~100 C++ |

nothing in PyTorch itself

## Risks

- Python hook costs tens of µs per op; small-kernel models drown in it. fix: C++ `RecordFunction`, or hook only ops that touch movable storages
- 2 MiB rounding per large tensor; activations churn, so the cache must work
- eager only. `torch.compile`, CUDA graph and NCCL memory stays put
- blocking moves: host waits last use, GPU queue drains, bubbles. 6C measures it, 6D fixes it
- pinned host memory cannot swap; `vmemd` host limit already caps it
- UVM + hints may simply win single-tenant (6C gate)

## Read before claiming anything (4E, unverified)

DeepUM (UVM + prefetch for DNN training), vDNN, SwapAdvisor, Capuchin (tensor swap in training), ZeRO-Offload and ZeRO-Infinity, FlexGen, Hugging Face `accelerate` offload hooks, torchtune activation offloading; MSched (4A), kvcached (Prism)
