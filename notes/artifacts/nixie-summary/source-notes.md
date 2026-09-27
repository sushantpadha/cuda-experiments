# Nixie: source notes (for Claude)

Raw reading notes behind `SUMMARY.md`. Paper = arXiv 2601.11743v1 (16 pages, read in full, local `references/nixie-paper.pdf`). Code = <https://github.com/XOR-op/Nixie>, commit `eebf583` (2026-09-24), cloned to session scratchpad, not vendored. Repo README: accepted at OSDI '26, pp. 2085-2101. Rust, about 13.3k lines (paper says about 10k).

## File map

| Path | What |
|---|---|
| `src/sidecar/` | the "Shim" of the paper: `libnixiesidecar.so`, loaded with `LD_PRELOAD` |
| `sidecar/src/intercept.rs` | `cudaMalloc`, `cudaFree`, `cudaMallocAsync`, `cudaFreeAsync`, `cudaMemcpy[Async]`, `cudaMemset`, `cudaMemGetInfo`, `cudaDeviceReset`, libc `open` |
| `sidecar/src/intercept_launch.rs` | `cudaLaunchKernel`, `cudaGraphLaunch`, `cudaStreamBeginCapture`/`EndCapture`, `cudaGraphInstantiate` |
| `sidecar/src/intercept_sync.rs` | `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventSynchronize` (timestamps only) |
| `sidecar/src/memory/mod.rs` | `cuMemCreate`/`cuMemMap`/`cuMemSetAccess`/`cuMemUnmap`/`cuMemRelease` wrappers |
| `sidecar/src/memory/streaming.rs` | per-device D2H and H2D streams; executes daemon migration RPCs inside the app process |
| `sidecar/src/schedule/mod.rs` | execution gate (mutex + condvar), idle monitor thread |
| `common/src/shm.rs` | per-process allocation table in POSIX shm, read by the daemon |
| `common/src/constant.rs` | 2 MiB min block, 128 MiB max chunk, 40960 handles, 8 GPUs |
| `daemon/src/runtime/schedule/policy.rs` | MLFQ queue, promotion/demotion, preemption test, auto-prefetch plan |
| `daemon/src/runtime/schedule/scheduler.rs` | event loop, context switch, `perform_migration` |
| `daemon/src/runtime/migration/migration_plan.rs` | global migration plan (victim selection, tier destinations) |
| `daemon/src/runtime/migration/execution.rs` | pipelined executor: D2H and H2D tasks, VRAM tokens, SHM back-pressure |
| `daemon/src/runtime/migration/channel.rs` | `ShmCoordinator`: GPU-to-SHM reservations take priority over backend-to-SHM |
| `daemon/src/runtime/migration/{shm,hostmem,storage}_buffer.rs` | the three off-GPU tiers |

## Facts from code

- Init: interposed libc `open()` calls `init_all_entrypoint()` on first call, connects to `/tmp/nixie.sock`. No daemon: `process::exit(1)`.
- Handshake sends pid, per-process shm path (`/nixie_ipc-<pid>-<uuid>.shm`), `CUDA_VISIBLE_DEVICES`. Daemon replies with VRAM limit per device (NVML, default 0.95 of device) and the global pinned buffer path (`/nixie_shm_buffer`).
- Pinned tier = one POSIX shm region created by the daemon, `mmap`ed by every app and `cuMemHostRegister(..., CU_MEMHOSTALLOC_PORTABLE)` in each app. Default 32 GiB.
- Host tier = daemon heap, 2 MiB `BytesMut` blocks, pageable, default 32 GiB plus burst. Optional prefault.
- Disk tier = one file `/tmp/nixie.pagebuffer`, truncated at start.
- `cudaMalloc(size < 2 MiB)`: real `cudaMalloc`, counted, never migrated.
- `cudaMalloc(size >= 2 MiB)`: round up to 2 MiB, `cuMemAddressReserve`, split into handles of at most 128 MiB (linked list in shm table), `cuMemCreate` each on device, `cuMemMap`, `cuMemSetAccess`. On OOM: pause and ask daemon for `remaining + 2 x 128 MiB`, then retry.
- `cudaFree`: unmap and release on GPU; if paused, tell daemon to drop off-GPU copies. `cuMemAddressFree` is never called anywhere in the repo: reserved VA leaks (harmless in practice, VA space is large). No driver-API hooks at all (`cuLaunchKernel`, `cuMemAlloc` absent).
- `cudaMallocAsync`: sync stream, then `cudaMalloc`; `cudaFreeAsync` caches blocks up to 256 MiB behind an event.
- `cudaMemGetInfo`: `free = limit - own usage`.
- `cudaStreamCreate` NOT hooked in this commit (paper §4 says it is). Instead `require_reserved_memory(256 MiB)` after malloc and graph instantiate; context creation reserves 768 MiB.
- Gate: `launch_allowed()` on kernel, graph launch, capture begin, malloc, memcpy, memset. If paused: send `RequestScheduling` (or `YieldThenRequestSchedulingAndMem` on OOM), wait on condvar.
- Disable RPC: state Paused, `cuCtxSynchronize` on every device. Enable RPC: state Running, notify.
- Idle monitor (every 20 ms, if `auto_idle`): idle when last kernel > 100 ms, graph > 200 ms, malloc > 300 ms, transfer > 300 ms (or small), blocking transfer > 100 ms, sync > 100 ms, no pending sync or blocking copy, and running > 100 ms. Sends `Idle`.
- Migration in the app: D2H = `cuMemcpyDtoHAsync` into SHM 2 MiB blocks on a non-blocking stream, worker thread waits the event, then `cuMemUnmap` + `cuMemRelease`. H2D = `cuMemCreate(full chunk)`, `cuMemMap` at the same VA, `cuMemcpyHtoDAsync` from SHM blocks. `todo!("Fallback")` on OOM.
- Chunk is the GPU unit (one handle, up to 128 MiB). Block (2 MiB) is the unit in SHM/host/disk; a chunk's blocks can be scattered.
- Planner: required = incoming footprint - (free VRAM - 2 x 128 MiB). Victims = other processes in daemon list order, entries largest first, stop when enough. Destination: SHM if free; else evict an SHM resident (HashMap order, arbitrary) to host mem if space, else disk. No LRU, no hotness, no access counters.
- Executor: D2H task and H2D task per device run concurrently. Each D2H completion returns a "token" with freed bytes; H2D waits until tokens cover its next chunk. Backend-to-SHM loads and SHM-to-backend spills run in `spawn_blocking` threads.
- `ShmCoordinator`: a pending GPU-to-SHM reservation blocks backend-to-SHM reservations. Closest code analog of the paper's "streaming window" (my reading).
- Scheduler: 5 levels (Interactive, LowInteractive, HighBatch, Batch, Background). Demotion quantum 8/16/32/64/128 s. Dynamic floor = Batch. Promotion rule in `update_priority`. Pop order: idle > prefetch > schedule (if front priority >= active; equal priority waits cooldown) > auto-prefetch for queue head.
- Cooldown after a switch = max(config, 3 x migration_MB / 16 GB/s, quantum of level). So 8 s at Interactive. `priority_level_to_cooldown` (4/8/16/32/64 s) is defined but unused.
- Lazy residency: idle app becomes `LastActive`, data stays on GPU until another app needs the space.
- MLFQ periodic reset commented out.

## Paper vs code

| Topic | Paper | Code (eebf583) |
|---|---|---|
| Idle threshold | 100 ms (§6.1) | 100 to 300 ms per API class |
| Preemption threshold S | 4 s top queue (§6.3) | cooldown = max(..., 8 s quantum); 4 s table unused |
| Promotion (Alg. 1) | `i_a - R*q_a > T_{p-1} + t_a`, `R < 1/N` | `idle - q*(m-1)/m > T/2 + t_a`, `m = max(N,2)+1`; keeps 1/m of queue time |
| Hooked APIs | includes `cudaStreamCreate` (§4) | not hooked; fixed reservations instead |
| Levels K | unspecified | 5 |
| Size | ~10k lines | ~13.3k lines |

## Open

- Does llama.cpp's default build link `libcudart` shared? Needed for the interposition to work. Not verified.
- PyTorch with `expandable_segments:True` calls `cuMemCreate` directly: bypasses Nixie. Not tested, inferred from the hook list.

## Streams and graphs (added 2026-09-27)

- Capture-begin hook is exported as `cudaStreamCaptureBegin` (`intercept_launch.rs:62`) but the CUDA API is `cudaStreamBeginCapture`. `dlsym` inside it looks up the right name, but apps never call the misnamed export. So `IS_DURING_CAPTURE` is never set, and the spin in `set_allow_running` (`schedule/mod.rs:81`) never waits. The §4 "CUDA graph compatibility" guarantee is not active at eebf583. Found by reading; not run.
- Even if fixed: one global bool for all streams; two overlapping captures clear it early.
- Disable during capture calls `cuCtxSynchronize`, which is prohibited during global-mode capture and invalidates it (CUDA Programming Guide, stream capture "prohibited and unhandled operations"; verify exact wording).
- `cudaMallocAsync` hook calls `cuStreamSynchronize(stream)`; on a capturing stream this is illegal, so stream-ordered allocation inside graphs breaks under Nixie. Inferred.
- Graph-owned memory (memory nodes, `cudaGraphAddMemAllocNode`) never goes through `cudaMalloc`: invisible to Nixie, not migrated. Inferred.
- Earlier line refs in the explorer's hook table (337/348/360/372) came from concatenated output; corrected to 51/62/74/86.
