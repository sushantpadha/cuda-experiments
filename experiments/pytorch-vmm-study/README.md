# VMM API usage in PyTorch's CUDA caching allocator

Study notes. Goal: understand exactly how PyTorch drives the CUDA driver
Virtual Memory Management (VMM) APIs so we can reuse the pattern in `VMMVector`.

All source line numbers are approximate and track `pytorch/pytorch` **main**.
The relevant code is one class, `ExpandableSegment`, in a single file.

---

## 1. The one file that matters

**`c10/cuda/CUDACachingAllocator.cpp`** — the whole caching allocator.
- <https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp>
- Class `ExpandableSegment` (~L1000–1500) is the entire VMM implementation.
- Everything else in the file is the block/segment bookkeeping layer that sits
  on top and is VMM-agnostic.

Header / enum plumbing:
- `c10/cuda/CUDACachingAllocator.h`
- `c10/cuda/CUDAAllocatorConfig.h` — parses `PYTORCH_CUDA_ALLOC_CONF`
  (`expandable_segments:True`, `expandable_segments_page_size:<bytes>`).

---

## 2. Two-layer model

| Layer | Unit | What it does |
|---|---|---|
| Caching allocator | **Segment** → **Block** | Finds a free block, splits oversized ones, merges adjacent frees. Blocks in different segments *never* merge. |
| `ExpandableSegment` | **VA reservation** → **page handles** | One huge `cuMemAddressReserve`, physical `cuMemCreate` handles mapped in page-sized chunks on demand. |

Pools (never share memory, even with expandable segments):
- **small**: allocations ≤ 1 MiB, 2 MiB page granularity
- **large**: allocations > 1 MiB, 20 MiB page granularity (configurable)

**Why expandable segments cut fragmentation:** without them every `cudaMalloc`
is its own segment and a 16 MiB free block can't combine with another to serve
32 MiB. With them, one segment owns one contiguous VA range, so frees merge
across old split boundaries; once everything in a segment is freed (common with
CUDA graphs) it consolidates to a single free region and allocation order stops
mattering.

- DevLog, *When does fragmentation occur in the CUDA caching allocator?*
  <https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/>
- Docs, *Memory management / Optimizing memory usage with `PYTORCH_CUDA_ALLOC_CONF`*
  <https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf>

---

## 3. `ExpandableSegment` — VMM call map

Compare each against `VMMVector/vmm_vector.cuh`.

### Construction — reserve VA once
- `cuMemAddressReserve` (~L1050–1080). Reserves ~device-memory + 1/8 of
  address space up front (256 TiB VA is free, physical is not). Stream-scoped
  variants reserve less.
- Same idea as `VMMVector` ctor `cuMemAddressReserve(&d_ptr, max_capacity_ * sizeof(T), ...)`
  — PyTorch just sizes it to the whole GPU instead of a caller-supplied cap.

### `map(range)` — commit physical pages
1. For each page index in the range: `cuMemCreate` a
   `CUmemGenericAllocationHandle` of exactly `segment_size` (~L1150–1180).
   This is the call that actually consumes GPU memory.
2. `mapAndSetAccess()`: `cuMemMap` the handle at `ptr + i*segment_size`, then
   `cuMemSetAccess` with a read-write `CUmemAccessDesc` (~L1340–1380).
- `VMMVector` does the same three calls in `push_chunk` / `map_chunk` /
  `set_access`, one chunk at a time in `reserve()`.

### `unmap(range)` — release physical, keep VA
- Sync the stream, then `unmapHandles()` (~L1390–1450): `cuMemUnmap` each page,
  optionally `close()` the exported fd, `cuMemRelease` the handle.
- VA stays reserved so the range can be re-mapped later.
- `VMMVector` mirrors this in `unmap_chunk` + `pop_chunk` (`cuMemUnmap` then
  `cuMemRelease`). PyTorch *always* releases the handle on unmap — see
  [issue #166116](https://github.com/pytorch/pytorch/issues/166116) for the
  argument that it should sometimes retain it (exactly what `retain_last_` does
  in `VMMVector`).

### Peer access
- `addPeer(dev)` → `setAccess()` re-runs `cuMemSetAccess` for every
  already-mapped range with a `CUmemAccessDesc` for the new device (~L1305–1345).
  `O(log N)` per allocation vs `cudaEnablePeerAccess`'s all-to-all mapping.

### Cross-process sharing — `share()` / `fromShared()`
- `share(range, stream)` (~L1195–1275): writes a `ShareHeader`
  (PID, `segment_size`, handle count, handle type) then
  `cuMemExportToShareableHandle` per page → POSIX fd or fabric handle.
- `fromShared(...)` (~L1280–1335): reads header, uses `pidfd_open` +
  `pidfd_getfd` to pull the producer's fds, `cuMemImportFromShareableHandle`
  per page, own `cuMemAddressReserve`, then `mapAndSetAccess`.
- Note: importing a handle re-reserves ~9/8 of device VA per import — see
  [issue #186213](https://github.com/pytorch/pytorch/issues/186213).
- `VMMVector` already sets `props_.requestedHandleTypes =
  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` — the export/import path is directly
  reusable for our multi-process design.

### Importing *external* VMM allocations
- Expandable segments can adopt handles created by other VMM allocators
  (e.g. `ncclMemAlloc`) via `cuMemRetainAllocationHandle`, then manage them in
  the normal segment lifecycle.
- [RFC #165419](https://github.com/pytorch/pytorch/issues/165419).

---

## 4. What PyTorch does *not* do (our opportunities)

- **No remap of backing between device and host.** Pages are device-only;
  `unmap` always frees. No "evict this page to `HOST_NUMA`, keep the VA".
- **No shared allocator state.** `share()` moves handles between processes, but
  each process runs its own independent block allocator over the imported VA.
  There is no cross-process free list / coordinator.
- **`empty_cache()`** unmaps pages but the DevLog doesn't claim physical pages
  return to the driver pool eagerly.

---

## 5. Background reading

- NVIDIA, *Introducing Low-Level GPU Virtual Memory Management* (the canonical
  intro, incl. the "growing vector" example):
  <https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/>
- CUDA Driver API, *Virtual Memory Management* (all `cuMem*` signatures):
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html>
- CUDA Programming Guide, *Extended GPU Memory (EGM)* — device↔host NUMA backing:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/egm.html>
- Bruce-Lee-LY, *Nvidia GPU Virtual Memory Management* (annotated walkthrough):
  <https://bruce-lee-ly.medium.com/nvidia-gpu-virtual-memory-management-7fdc4122226b>
- GMLake (ASPLOS'24) — VMM stitching for defrag, similar mechanism:
  <https://arxiv.org/pdf/2401.08156>
- vTensor / vLLM — VMM for KV-cache:
  <https://arxiv.org/pdf/2407.15309>
- antgroup/glake — production VMM allocator, compares itself to `expandable_segments`:
  <https://github.com/antgroup/glake>
