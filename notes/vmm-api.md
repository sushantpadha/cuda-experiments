# CUDA VMM API, compared with cudaMalloc and cudaMallocManaged

> AI-generated :) notes for the background section

Main source: Perry and Sakharnykh, [Introducing Low-Level GPU Virtual Memory Management](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/), NVIDIA blog, April 2020 (CUDA 10.2). The post is old, but the mechanism it describes is unchanged in CUDA 13. Current API docs:

- Driver API, virtual memory management: <https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html>
- Programming Guide, Virtual Memory Management: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html>
- Programming Guide, Unified Memory: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html>
- Programming Guide, Stream-Ordered Memory Allocator: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html>
- Runtime API, memory management (`cudaMalloc`, `cudaFree`, `cudaMallocManaged`): <https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html>
- Runtime API, synchronization behavior: <https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html>

## The mechanism

`cudaMalloc` does three things in one call: it picks a virtual address range, allocates physical memory, and maps one onto the other. VMM splits these into separate driver calls, each with its own inverse:

| Step | Call | Inverse |
|---|---|---|
| reserve a virtual address range, no memory behind it | `cuMemAddressReserve` | `cuMemAddressFree` |
| allocate physical memory, returns a handle | `cuMemCreate` | `cuMemRelease` |
| map a handle onto part of a reserved range | `cuMemMap` | `cuMemUnmap` |
| grant read or read/write access per device | `cuMemSetAccess` | (reset by `cuMemUnmap`) |

Rules that follow from the split:

- **Granularity.** `cuMemCreate` sizes must be multiples of `cuMemGetAllocationGranularity` (2 MiB on our GPU).
- **No partial mapping.** `cuMemMap` maps a whole handle; the size must equal the handle's size.
- **No access by default.** A new mapping is `PROT_NONE` for every device, so a kernel that touches it faults until `cuMemSetAccess` is called. This is needed again after every `cuMemMap`, including a remap at the same address.
- **Reference counting.** `cuMemRelease` may be called while the handle is still mapped; the memory is freed only when the last mapping is removed and every reference (including exported handles) is released.
- **Address hint.** `cuMemAddressReserve` takes an optional fixed address. If it cannot be used, the driver picks another address instead of failing, so the caller must check the result.
- **Sharing.** A handle created as exportable can be turned into an OS handle (`cuMemExportToShareableHandle`; a file descriptor on Linux) and imported in another process. The legacy `cudaIpc*` functions do not work on VMM memory.

The blog says only pinned device memory could be created. Later releases added more locations; this project uses pinned host memory on a NUMA node (`CU_MEM_LOCATION_TYPE_HOST_NUMA`), which kernels read over PCIe.

## Contrast with cudaMalloc

- **Growing a buffer.** With `cudaMalloc`, growth means allocate new, copy, free old, so a buffer cannot grow past half of free memory, and every growth step costs a copy. With VMM, reserve a large range once and map new chunks at its end. If the range cannot be extended, unmap the old handles and map them into a larger range without copying data.
- **Free synchronizes.** `cudaFree` waits for all pending work on the current context and on its peer contexts before it returns. Other threads' kernels stall behind an unrelated free; VMM teardown does not (see below).
- **Memory use.** `cudaMalloc` commits everything it reserves. VMM commits only the chunks that are mapped.

## Contrast with cudaMallocManaged (UVM)

- **Commit on demand.** On systems with concurrent managed access (Linux on x86 with a recent GPU, as here), `cudaMallocManaged` returns before any memory is populated; pages are created and migrated when touched, and the driver may evict them to host memory at any time. VMM never faults or migrates: memory is pinned where `cuMemCreate` put it, and moving it is an explicit unmap, copy and map.
- **Who decides placement.** UVM placement is a device-wide driver policy, steered only by hints (`cudaMemAdvise`, `cudaMemPrefetchAsync`). With VMM the application decides.

## Which calls synchronize

"Synchronous" below means the host call blocks until the operation completes. "Device-synchronizing" means it also waits for previously launched GPU work.

| Call | Host blocks | Waits for in-flight GPU work | Source |
|---|---|---|---|
| `cudaMalloc` | yes | not documented | runtime docs |
| `cudaFree` (memory not from `cudaMallocAsync`) | yes | **yes**, current context and its peers | blog; runtime docs exempt only `cudaMallocAsync` memory |
| `cudaMallocManaged` | yes, returns before pages exist | not documented | runtime docs |
| `cudaMallocAsync`, `cudaFreeAsync` | no, stream-ordered | ordered in the stream only | Stream-Ordered Memory Allocator |
| `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemRelease`, `cuMemAddressFree` | yes (no stream argument) | not documented; did not wait in our runs | driver docs; observation below |
| `cuMemUnmap`, `cuMemSetAccess` | yes, "exhibits synchronous behavior for most use cases" | **not guaranteed**; may on Maxwell or older | blog; driver docs |

What this means in practice:

- Every VMM call is a blocking host call with no stream argument. There is no stream-ordered map or unmap for ordinary buffers.
- VMM calls do not wait for kernels. The caller must make sure no kernel is using a range before unmapping it, for example with a stream or event synchronize on the streams that use it. Nothing in the driver catches a mistake: the kernel faults.
- The runtime docs add that any CUDA call may block for internal reasons, such as contention for resources, and that this undocumented behaviour should not be relied on.

## This project's preliminary observations (RTX 4050, CUDA 13.0)

- A 64 MiB remap (create, copy, unmap, map, set access) took about 16 ms while an unrelated 450 ms kernel kept running, so these calls did not wait for that kernel.
- Unmapping a buffer while a kernel reads it crashes the kernel, consistent with "no wait".
- `cuMemSetAccess` and `cuMemUnmap` on device memory sometimes take about 2 ms instead of tens of µs, mostly after a large device free. The likely cause is the driver clearing freed VRAM; not confirmed.
- `cudaFree` and VMM unmap + release cost the same from 2 MiB to 4 GiB (about 1.6 ms at 4 GiB), and `cudaMalloc` and VMM create + map + set access cost about the same (0.2 to 0.6 ms). `cudaMallocManaged` alone costs about 60 µs, but backing its pages by GPU faults costs about 0.15 ms per MiB.

Caveats: the blog describes 2020 hardware and drivers. The "host blocks" column for calls the docs do not describe is based on the API shape (no stream argument) and our timings.
