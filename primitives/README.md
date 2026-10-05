# libvmem v0 primitives

**Status: partial, work in progress.** Tracker 1A. One header, `vmem.cuh`: a `vmem::Manager` that wraps the CUDA VMM driver API in blocking, error-checked calls.

## v0 subgoals (tracker 1A)

- [x] Primitives in `vmem.cuh`: reserve, create (device or pinned host), map, unmap, remap to a chosen location, release, free; residency queries; chunk tokens
- [x] `smoke.cu`: every call, data checked across remaps, error paths
- [~] `main.cu`: first timing test (remap vs. `cudaMemcpyAsync` to host); under review
- [ ] Review `vmem.cuh` (API, remap path, destructor, error handling)
- [ ] Action runner (1Ai): read a simple action file (reserve, create, map, remap, unmap, release, free, plus a kernel touch/check) and execute it
- [ ] Parallel runs (1Aii): many runner processes at once; check correctness and basic speed

Not in v0: threads, async remap, hooks, policies.

## Open issues to review

- `main.cu` calls `remap` right after launching `fill` on a non-blocking stream. `remap`'s copy is ordered only after the legacy default stream, so it can race the kernel; sync the stream first.
- `main.cu`'s `check` kernel writes to a host stack variable (`&bad`). That works here only because the driver has HMM enabled; use managed or device memory to be portable.
- `remap` is blocking; the copy could run async with the swap done once it lands.

## API

This is a **per-process** utility for exercising VMM: one `Manager` lives inside one process and owns only that process's memory. It is not a shared allocator or a daemon.

| Call | Does | Returns |
|---|---|---|
| `Manager(Options)` | `cuInit`, primary context, checks VMM and host-NUMA support, granularity | |
| `reserve(size)` | reserve a virtual range | base VA |
| `create(size, Loc::Device \| Loc::Host)` | physical memory on the GPU or in pinned host memory | `ChunkToken` |
| `map(tok, va)` | map inside a reservation, grant device read/write | `va` |
| `unmap(tok)` | unmap from its stored address | that address |
| `remap(tok, Loc)` | move a mapped chunk to the given location, same VA, data copied; prints and does nothing if already there | its address |
| `release(tok)` | free physical memory (must be unmapped) | chunk size |
| `va(tok)`, `ptr<T>(tok)` | mapped address (stable across remap), e.g. for kernel arguments | `CUdeviceptr`, `T*` |
| `loc(tok)`, `on_device(tok)` | residency right now | `Loc`, `bool` |
| `info(tok)` | snapshot: id, size, location, address | `ChunkInfo` |
| `free(va, size)` | free a reservation (nothing mapped inside) | size |
| `print_state()` | reservations, chunks, VRAM free | |

Chunks are referred to by `ChunkToken`, a copyable value. Tokens are never reused, so a token kept after `release` throws instead of naming a newer chunk; a token from another `Manager` throws too.

Sizes round up to the granularity (2 MiB here). Failures throw `vmem::Error` (`.code` holds the `CUresult` when a CUDA call failed). `Options{device, host_numa, verbose, debug}`: `device` picks the GPU (default 0), `host_numa` the NUMA node for host chunks (default -1 = the node closest to that GPU, else 0); `verbose` prints one line per call, `debug` also dumps the state after each call. The destructor unmaps, releases and frees whatever is left.

Rules: single-threaded for now (a mutex-protected version comes later); never `unmap` or `remap` a chunk while a kernel may touch it (crash, no fault handler); `remap`'s copy is ordered after the legacy default stream only.

```
make            # builds ./smoke and ./main
./smoke         # every call, data checked across remaps, error paths
./smoke -d      # same with state dumps
./main 12 24    # timing test: remap 12 MiB to host vs. memcpy 24 MiB
make debug      # -g -G -DDEBUG build
```

## Toward a daemon model

What a later shim + daemon design will need on top of this:

- one `Manager` per tenant process: only the owning process can map its own addresses
- a command channel so the daemon can ask a process to remap or release a chunk (tokens are plain integers, so they travel over IPC)
- thread safety: commands arrive on another thread
- knowing when a chunk is idle, so a remap never races a running kernel
- a host-memory budget across processes (MPS limits cover device memory only)
- handle export (`cuMemExportToShareableHandle`) if chunks are shared between processes
