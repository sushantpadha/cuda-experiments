# How an NVIDIA GPU runs your work

Scheduling parts from Bakita and Anderson, *Demystifying NVIDIA GPU Internals*, RTAS 2024. VMM parts from the CUDA Programming Guide and this project's runs (the paper does not cover memory).

## The pieces

A GPU is a set of independent **engines**: one compute engine (all the SMs, runs kernels), several **copy engines** (move data between GPU and CPU memory), plus video and JPEG units. They can all work at the same time.

| Word | What it really is |
|---|---|
| **Context** | one program's private GPU state on one GPU: its address space (page tables), loaded kernels, streams. Usually one per program |
| **Stream** | an ordered queue of work (kernels, copies) inside a context |
| **Channel** | the hardware queue a stream is written into: a command buffer in memory that the GPU reads |
| **TSG** (time-slice group) | all channels of one context, bundled. The unit the GPU switches between |
| **Runlist** | the hardware's list of TSGs waiting to run on a set of engines |

## A kernel launch, end to end

```
kernel<<<...>>>  ->  CUDA library writes commands into a channel's buffer (no system call)
                 ->  the GPU scheduler picks a TSG from the runlist, then a channel inside it
                 ->  a fetch unit pulls the commands and hands them to the compute engine
```

Launching is cheap because the program writes straight into memory the GPU reads. The kernel driver is only involved at setup.

## The rules that matter

1. **One context at a time per engine.** The scheduler gives each TSG (each program) a time slice, about 2 ms, round-robin. Two programs that each use a quarter of the GPU still take turns. That is why two processes without MPS each ran 2x slower in our test.
2. **Copies and kernels from different programs can overlap.** Compute and copy engines usually sit on separate runlists, so program A's copy can run during program B's kernel.
3. **Only 8 channels per context by default.** A 9th stream has to wait for a free channel even if the GPU is idle, so streams can stall each other for no visible reason. `CUDA_DEVICE_MAX_CONNECTIONS` raises the limit.
4. **Copy engines can secretly share hardware.** CUDA's "copy engines" are logical; two of them can map to one physical engine and slow each other (seen on Ada GPUs, our RTX 4050's family; not checked here).
5. **Everything goes through channels**, even device-mapped allocations. Freeze a program's channels and it cannot launch, copy or allocate.

## MPS

MPS merges many processes into one context (each becomes a sub-context). One context means one TSG, so their kernels really run at the same time. The price: they share fate, so one client's crash kills the others (we saw this).

## VMM (virtual memory management)

Normal `cudaMalloc` hands you memory and an address together. VMM splits that into steps you control:

| Step | Call | What happens underneath | Cost here (RTX 4050) |
|---|---|---|---|
| reserve | `cuMemAddressReserve` | claims a range of addresses in the context; nothing behind it | ~3 µs |
| create | `cuMemCreate` | allocates physical memory (VRAM, or pinned host RAM) | VRAM ~66 µs, any size; host ~107 µs per MiB |
| map | `cuMemMap` | records "this address range uses this memory" | ~3 µs |
| set access | `cuMemSetAccess` | writes the GPU page tables so kernels may touch it | VRAM ~70 µs flat; host ~8.6 µs per MiB |
| unmap / release / free | | the reverse | tens of µs (release grows with size) |

Things to know:

- **Page tables live in the context.** Mapping is per process; another process can use the same memory only by importing a handle (a file descriptor) and mapping it itself.
- **VRAM is cheap to map, host RAM is not.** Likely reason (our inference): VRAM comes in big contiguous pieces, so a few large page-table entries cover a lot, while pinned host RAM is scattered 4 KB pages, so every MiB needs more entries and more pinning work.
- **No page faults.** Touching an address that is not mapped kills the context. VMM never moves data for you; you copy and remap yourself.
- **Host-backed memory works from kernels**, over PCIe: about 13 GB/s here vs ~187 GB/s from VRAM.

## Glossary

| Term | Meaning |
|---|---|
| SM | a block of cores inside the compute engine |
| Engine | an independent GPU unit: compute, copy, video |
| Page table | the GPU's map from addresses to physical memory |
| Pinned memory | host RAM the OS may not move or swap, so the GPU can read it directly |
| `nvdebug` | the paper's kernel module that shows live channels and runlists (`/proc/gpuX/runlistY`) |
