# Nixie: how it works

Xu, Wang, Ren, Chen, Zhuo (Duke). *Nixie: Efficient, Transparent Temporal Multiplexing for Consumer GPUs.* arXiv 2601.11743v1, Jan 2026; OSDI '26 per the repo README. Code: <https://github.com/XOR-op/Nixie> (Rust, read at commit `eebf583`, 2026-09-24).

Section numbers (§) and bracketed numbers ([n]) refer to the paper and its bibliography. Code paths are relative to the repo's `src/`. Where paper and code differ, both are given; the table in part 7 collects them. Raw notes: `source-notes.md`.

## 0. The one idea

A consumer GPU runs a few large apps (llama.cpp [13], SGLang [42], ComfyUI [9], Ollama [25]) whose working sets each nearly fill VRAM (§2.1). Nixie gives the GPU to **one app at a time** and makes sure that app's **entire** footprint is in VRAM before its kernels run. Switching apps = stop the old one, move its memory out, move the new one's in. No page faults, so no UVM thrash (§1, §3).

Everything else serves that: interposition to see and control allocations and launches (part 1), a tiered off-GPU store (part 2), a fast planned migration (part 3), a scheduler that decides who gets the GPU and for how long (parts 4, 5).

```
                ┌──────────────────────── nixie daemon (one per machine) ───────────────────────┐
                │  Scheduler (MLFQ)          Migration planner + executor                      │
                │  policy.rs, scheduler.rs   migration_plan.rs, execution.rs                   │
                │        ▲  │                     │                    │                        │
                │        │  │ Enable/Disable       │ migrate RPC         │ SHM ⇄ host ⇄ disk      │
                └────────┼──┼─────────────────────┼────────────────────┼────────────────────────┘
      UNIX socket (tarpc)│  │                     │                    │
   /tmp/nixie.sock       │  ▼                     ▼                    │
┌─ app process 1 ────────┴──────────────┐ ┌─ app process 2 ───────────┐│
│ app code (llama.cpp, torch, ...)      │ │ app code                  ││
│   │ cudaMalloc / cudaLaunchKernel ... │ │                           ││
│   ▼                                   │ │                           ││
│ libnixiesidecar.so  (LD_PRELOAD)      │ │ libnixiesidecar.so        ││
│   gate · VMM mapping · copy streams   │ │                           ││
│   ▼  (real symbols via dlsym(RTLD_NEXT))                           ││
│ libcudart.so → libcuda.so → driver    │ │ own CUDA context          ││
└───────────┬───────────────────────────┘ └─────────┬─────────────────┘│
            │ cuMemCreate/Map (VRAM)                │                  │
            ▼                                       ▼                  ▼
   ┌──────────── GPU VRAM ─────────────┐   ┌── /nixie_shm_buffer: POSIX shm, pinned ──┐
   │ only the active app's chunks      │⇄⇄⇄│ cuMemHostRegister'd in every app          │
   └───────────────────────────────────┘   └───────────────────────────────────────────┘
                                                  ⇅ daemon memcpy       ⇅ daemon file I/O
                                           daemon heap (pageable)   /tmp/nixie.pagebuffer
```

Figure 2 in the paper is the same picture with steps ①–⑥; part 3 walks through them against the code.

## 1. Interposition: how `LD_PRELOAD` gets it in (§4)

### 1.1 The mechanism

`LD_PRELOAD=/path/lib.so ./app` tells the Linux dynamic linker to load `lib.so` before every other shared library. When the app calls a function that is resolved dynamically (for example `cudaMalloc` exported by `libcudart.so`), the linker finds the first definition in load order, which is now Nixie's. Nixie's function can call the real one by asking the linker for the *next* definition: `dlsym(RTLD_NEXT, "cudaMalloc")`.

No driver change, no app recompile. The driver still sees an ordinary CUDA process with its own context; Nixie only sits between the app and `libcudart.so`.

`nixie run <cmd>` just sets `LD_PRELOAD=libnixiesidecar.so` (and `CUDA_VISIBLE_DEVICES` if `-d` is given) and execs the command (`docs/cli.md`).

Every hook in the sidecar has the same shape (`sidecar/src/intercept.rs`, trimmed):

```rust
macro_rules! generate_init_fn {               // look up the real symbol once
    ($ty:ty, $name:expr) => {
        fn init_fn() -> $ty {
            unsafe { std::mem::transmute(dlsym(RTLD_NEXT, $name.as_ptr())) }
        }
    };
}

#[unsafe(no_mangle)]                          // export under the exact C name
pub extern "C" fn cudaLaunchKernel(func: *const c_void, grid: CudaDim3, block: CudaDim3,
                                   args: *mut *mut c_void, shmem: usize,
                                   stream: CUstream) -> cudaError_enum {
    static REAL: OnceLock<CudaLaunchKernelType> = OnceLock::new();
    generate_init_fn!(CudaLaunchKernelType, cr"cudaLaunchKernel");
    let real = REAL.get_or_init(init_fn);
    SCHED_CTL.launch_allowed(LaunchType::Kernel);   // ← blocks here if this app is paused
    real(func, grid, block, args, shmem, stream)
}
```

(`sidecar/src/intercept_launch.rs:24-48`.)

### 1.2 What is hooked

From the `extern "C"` definitions in `sidecar/src/` at `eebf583`:

| Group | Symbols | What Nixie does |
|---|---|---|
| Allocation | `cudaMalloc`, `cudaFree`, `cudaMallocAsync`, `cudaFreeAsync` | replace with VMM (part 2.2) |
| Launch | `cudaLaunchKernel`, `cudaGraphLaunch`, `cudaStreamBeginCapture` (exported by mistake as `cudaStreamCaptureBegin`, so never reached), `cudaStreamEndCapture`, `cudaGraphInstantiate` | gate: block while paused; timestamp |
| Transfer | `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemset` | gate; timestamp; set the right context for VMM memory |
| Sync | `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventSynchronize` | timestamp before and after (idle detection, part 5) |
| Info | `cudaMemGetInfo` | lie: `free = limit − this app's usage` (§4 "GPU Memory information") |
| Misc | `cudaDeviceReset`; libc `open` | reset bookkeeping; first `open()` connects to the daemon |

Startup: the first call to libc `open()` (any path) runs `init_all_entrypoint()`, which connects to `/tmp/nixie.sock`, creates a per-process shared-memory table `/nixie_ipc-<pid>-<uuid>.shm`, and handshakes. The daemon answers with the VRAM limit (NVML, default 95% of the device) and the name of the global pinned buffer (`sidecar/src/comm/init.rs`, `sidecar/src/intercept.rs:559-569`).

### 1.3 Constraints this puts on applications

The paper says "any unmodified CUDA application" (§1, §4). The mechanism implies these limits. Items marked *inferred* follow from the hook list and linker rules; they are not stated in the paper or tested here.

1. **Must reach CUDA through a shared `libcudart.so`.** `LD_PRELOAD` only overrides dynamically resolved symbols. `nvcc` links the runtime statically by default (`-cudart static`); such a binary calls its private copy and never sees Nixie. PyTorch ships `libcudart.so`. *Inferred.*
2. **Runtime API only.** No driver-API symbol is hooked (`cuMemAlloc`, `cuMemCreate`, `cuLaunchKernel` are absent). Memory allocated through the driver API is invisible, never migrated, and still occupies VRAM when the app is paused. PyTorch with `expandable_segments:True` allocates with `cuMemCreate` directly. `cudaMallocManaged`, `cudaMallocHost`, `cudaHostAlloc` are not hooked either. *Inferred.*
3. **Allocations under 2 MiB stay on the GPU forever.** They go to the real `cudaMalloc` and are never migrated (`intercept.rs:111-129`).
4. **The daemon must be running.** If `/tmp/nixie.sock` is missing the sidecar calls `process::exit(1)` (`comm/init.rs`).
5. **Same user.** All apps map the same pinned buffer; any process with the same UID can read other apps' spilled data (§8 "Security").
6. **Stop-the-world pauses.** While paused, any thread of the app that calls a gated API blocks, including `cudaMemcpy` and `cudaMalloc`. CPU-only threads continue.
7. **Fixed tables.** 40 960 handles and 8 192 allocations per device, 8 GPUs (`common/src/constant.rs`, `common/src/shm.rs`).
8. **`cudaMemGetInfo` is virtualised.** Apps that size caches from it (SGLang sizes its KV cache this way, §4) see only their share.
9. **`cudaMallocAsync` loses its stream-ordered semantics.** It synchronises the stream and falls back to `cudaMalloc`; frees up to 256 MiB are cached behind an event (`intercept.rs:309-418`).
10. **Linux only** (Unix socket, POSIX shm, `LD_PRELOAD`). §8 argues a Windows port is feasible because VMM exists there [23].

## 2. Physical backings (tiers) and when each is used (§5.1, §5.2)

### 2.1 Units

- **Chunk** = one `cudaMalloc` of ≥ 2 MiB, rounded up to 2 MiB, split into pieces of at most **128 MiB** (§5.1). In code each piece is one `cuMemCreate` handle (`PhysicalMemoryHandle`, linked list per allocation, `common/src/shm.rs`). The piece is the unit of GPU residency: it is on the GPU entirely or not at all.
- **Block** = **2 MiB**, the VMM minimum granularity on their GPUs. Off the GPU a chunk is stored as up to 64 blocks that need not be contiguous. This stops fragmentation of the host tiers (§5.1).

Why 128 MiB: large enough to amortise per-handle cost, small enough that "roughly half resides in CPU pinned memory while the remainder is placed in CPU paged memory" is possible for a 256 MiB allocation (§5.1). Why not smaller: a VMM handle cannot be mapped partially (our own observation too), so smaller pieces mean more handles and more `cuMemMap` calls.

### 2.2 How an allocation is backed (VRAM)

`cudaMalloc(size ≥ 2 MiB)` (`sidecar/src/intercept.rs:131-211`, `memory/mod.rs:58-153`):

```rust
let rounded = (size + 2MiB - 1) & !(2MiB - 1);
cuMemAddressReserve(dev_ptr, rounded, 2MiB, 0, 0);           // fixed VA for the app's lifetime
while remaining > 0 {                                        // split into ≤128 MiB handles
    let len = remaining.min(128MiB);
    table.handle_list.allocate_handle(cur_addr, len);        // bookkeeping in shared memory
    ...
}
for h in handles {
    cuMemCreate(&mut h.cu_handle, h.size, &prop_device(dev), 0);   // physical VRAM
    // on OOM: pause, ask daemon for (remaining + 2×128 MiB), retry once
}
for h in handles {
    cuMemMap(h.addr, h.size, 0, h.cu_handle, 0);
    cuMemSetAccess(h.addr, h.size, &rw(dev), 1);
}
```

The pointer returned to the app is the reserved VA. It never changes, even after the physical memory behind it has moved to the host and back (§4 "Maintaining consistent virtual memory addresses", [24]).

The allocation table lives in a POSIX shared-memory file created by the sidecar and opened by the daemon, so the daemon can read every process's handles, sizes and `on_gpu` flags directly (`common/src/shm.rs`, `sidecar/src/lib.rs:65-104`).

### 2.3 The four tiers

| Tier | What it is in code | Default size | Pinned? | Who touches it |
|---|---|---|---|---|
| **GPU** | `cuMemCreate` handles mapped into each app's reserved VA | 95% of VRAM (NVML) | device memory | app (sidecar) only |
| **CPU pinned ("SHM")** | one POSIX shm region `/nixie_shm_buffer`, `mmap`ed by the daemon and every app, `cuMemHostRegister(..., PORTABLE)` in each app | 32 GiB (`--shmem`) | yes | daemon and all apps |
| **CPU paged ("hostmem")** | daemon heap, a pool of 2 MiB `BytesMut` buffers | 32 GiB (`--hostmem`) plus burst | no | daemon only |
| **Disk ("storage")** | one file `/tmp/nixie.pagebuffer`, truncated at start | grows | no | daemon only |

(`common/src/shm_buffer.rs`, `sidecar/src/init.rs:110-136`, `daemon/src/runtime/migration/{shm,hostmem,storage}_buffer.rs`, `daemon/src/runtime/daemon.rs:58-112`.)

Key rule, **exactly one copy** (§5.2): a chunk is in exactly one tier. When it moves up, the lower copy is freed. Nixie contrasts this with UVM: the paper asserts that UVM "allocates CPU pinned memory inside the kernel for DMA operations" and that "every page resident on the GPU must have a corresponding pinned page on the CPU" (§2.2), so three 24 GB apps would need 72 GB pinned (§5.2). The paper gives no citation or driver reference for this. **Checked against driver source 580.178.04** (`notes/artifacts/uvm-driver/README.md`): UVM moves the data, but keeps the old CPU page allocated as a "cached chunk" (kernel-owned, unswappable) until `cudaFree`. Pages first touched on the GPU have no CPU page, so "every page" overstates it; for CPU-loaded weights the claim holds. Not yet tested by a run. Nixie's measured result stands independently: same latency with 33-40% of UVM's pinned memory (Fig. 9).

The pinned tier is the only tier the GPU copies to or from. Paged host memory and disk are behind it: data always passes through SHM on its way to or from VRAM.

### 2.4 Criteria for choosing a tier

There is no per-chunk hotness. The criteria are whole-process and capacity-based:

- **GPU:** the chunks of the app that currently holds the GPU. Its whole footprint is brought in before it runs (§3).
- **Idle app, nobody waiting:** stays on the GPU ("lazy" residency). When its idle timer fires it becomes `LastActive` and keeps its VRAM until someone else needs the space (`scheduler.rs`, `handle_activity_idle`).
- **Eviction destination:** SHM if it has free blocks; otherwise make room in SHM by pushing some other SHM resident down to paged host memory if that has room, else to disk (`migration_plan.rs:280-328`, comment "Use SHM first, then host mem, then storage").
- **Which chunks leave the GPU:** only as many bytes as the incoming app is short by. Victims are the other processes in the daemon's client-list order, and within a process the largest chunks first (`migration_plan.rs:252-348`). Which SHM resident gets pushed down is the iteration order of a `HashMap`, which is arbitrary.
- **Prefetch:** when the GPU is busy and an app is at the head of the queue, its data is moved up the tiers (disk → SHM, host → SHM, disk → host) as far as capacity allows (`policy.rs:336-389`, §6.3).

So "which data is hot" is answered at process granularity by the scheduler (who runs next), not by the memory system.

## 3. Changing backings: when and how (§4, §5.3)

### 3.1 Triggers

1. **Context switch.** The scheduler picks a different app (part 4).
2. **Allocation OOM.** The running app's `cuMemCreate` fails; the sidecar pauses itself and sends `YieldThenRequestSchedulingAndMem` with the bytes it needs. The daemon evicts other apps' chunks to make that much room (`memory/mod.rs:71-97`, `utils.rs:52-78`).
3. **Prefetch.** Automatic for the queue head, or manual via `nixie prefetch 'pid:hostmem->shm=1g'` (`docs/cli.md`). Prefetch never touches the GPU of the running app.
4. **Free while paused.** `cudaFree` of a chunk that is off the GPU tells the daemon to drop its blocks.

### 3.2 A context switch, step by step

Paper Figure 2 steps ①–⑥ mapped to code. App 1 wants the GPU; App 2 holds it.

```
 App 1 sidecar            daemon                              App 2 sidecar
 ─────────────            ──────                              ─────────────
 cudaLaunchKernel()
  launch_allowed(): paused ①
  send RequestScheduling ───▶ schedule_push()  ②
  wait on condvar            poll_queue → schedule_pop()
                             (priority / cooldown test, part 4)
                             handle_sched_request()
                               Disable ──────────────────────▶ set Paused ③
                                                               cuCtxSynchronize() all devs
                               ListProcessResidual(App 1)      (running kernels finish)
                               ◀── wait for Disable ack ─────  ack
                               collect_all_residuals(others)
                               realtime_migrate_task() = plan
                               task.run()  ④ ─── migrate(D2H) ─▶ copy chunk → SHM, unmap, release
  ◀── migrate(H2D) ───────────  (pipelined, 3.3)
  cuMemCreate, cuMemMap,
  copy SHM → chunk
                               Enable ────▶ ⑤
  set Running, notify ⑥
  real cudaLaunchKernel()
```

(`sidecar/src/schedule/mod.rs:78-192`, `daemon/src/runtime/schedule/scheduler.rs:250-367, 531-631`.)

Safety comes from step ③: before any of App 2's memory is unmapped, App 2 is blocked from launching and `cuCtxSynchronize()` has drained its in-flight kernels (§4 "Valid memory access during migration"). This is the answer to "VMM ranges have no fault handler": the app is simply never running while its memory is gone. During CUDA graph capture the switch waits until capture ends, because an extra API call would break capture (§4 "CUDA graph compatibility"; `set_allow_running` spins on `is_during_capture()`).

Note that the copy is done **inside the app processes**, not by the daemon: VMM handles belong to the app's context, so only the app can `cuMemMap` or copy them. The daemon only sends `migrate` RPCs (§3 "each Nixie Shim migrates its own memory blocks").

The sidecar side of one migrate RPC (`sidecar/src/memory/streaming.rs:125-262`, trimmed):

```rust
if args.host_to_device {
    cuMemCreate(&mut h, total_size, &prop_device(dev), 0);   // new physical chunk
    cuMemMap(handle.addr, total_size, 0, h, 0);              // same VA as before
    cuMemSetAccess(handle.addr, total_size, &rw(dev), 1);
    for (off, sz) in args.blocks { cuMemcpyHtoDAsync(va + acc, shm_base + off, sz, h2d_stream); }
    cuEventRecord(ev, h2d_stream);
} else {
    for (off, sz) in args.blocks { cuMemcpyDtoHAsync(shm_base + off, va + acc, sz, d2h_stream); }
    cuEventRecord(ev, d2h_stream);
}
// a worker thread per direction: cuEventSynchronize(ev);
//   if D2H: cuMemUnmap + cuMemRelease   (VRAM freed only after the copy lands)
//   reply MigrationResponse::Success { size }  → becomes a "VRAM token" in the daemon
```

Two non-blocking streams per device, one per direction, so both copy engines can work at once.

### 3.3 Planning and pipelining (§5.3, Fig. 4, Fig. 5)

A naive design moves data tier by tier with back-pressure: VRAM waits for SHM, SHM waits for host, host waits for disk. The chain is long and can deadlock when the incoming data fills a tier that the outgoing data needs (§5.3).

Nixie does two things:

**1. One global plan up front** (`realtime_migrate_task`, `migration_plan.rs:129-427`). Inputs: the incoming app's chunks and where each is; every other app's GPU-resident chunks; free blocks in SHM and host. Output, for each device:

- `out_of_gpu`: which victim chunks leave VRAM (enough bytes, largest first);
- `hostmem_to_shm`, `storage_to_shm`: incoming chunks that must first be lifted into SHM;
- `shm_to_backend`: which current SHM residents must be pushed down to make room, and whether each goes to host or disk.

Fig. 5 in the paper shows this plan as a table of moves (GPU → pinned, GPU → pinned → paged, pinned → GPU, ...).

**2. A pipelined executor** (`DataMigrationTask::run`, `execution.rs:257-545`). Per device, a D2H task and an H2D task run concurrently:

```
D2H task (victims)                                  H2D task (incoming app)
for chunk in out_of_gpu:                            for chunk in incoming (ready-in-SHM first,
  blocks = reserve_from_gpu(chunk)  ◀── may wait      then as backend→SHM loads finish):
           for SHM space                              wait until tokens ≥ chunk.size
  rpc migrate(D2H) to victim app                      rpc migrate(H2D) to incoming app
  on reply: send token(size) ────────────────────▶   tokens -= chunk.size
            send "in SHM" to spill worker

SHM spill worker: SHM → host (else disk), in threads, frees SHM blocks for more D2H
Loader:           disk/host → SHM for incoming chunks not yet in SHM
```

- **Tokens** (`gpu_mem_token_tx/rx`): H2D may create a new VRAM chunk only after D2H has released at least that many bytes. That keeps VRAM from overflowing without waiting for the whole eviction to finish.
- **Full duplex**: GPU→host and host→GPU copies overlap, so both PCIe directions are busy. UVM's fault path evicts, then fetches, one direction at a time (§2.2, Fig. 1, Fig. 4). Measured: about 20 GB/s vs about 10 GB/s for UVM on a PCIe 5.0 x8 RTX 5090, against 21.5 GB/s from `nvbandwidth` [22] (Fig. 7).
- **Streaming window / priority** (§5.3): the paper reserves a small window of pinned memory for the evicting app so the D2H side never starves. In code the closest mechanism is `ShmCoordinator` (`channel.rs:223-334`): while any GPU→SHM reservation is pending, backend→SHM reservations wait. *My reading; the paper gives no code-level description.*
- **Multi-threaded host copies**: SHM ⇄ host and host ⇄ disk run in `spawn_blocking` threads to saturate DRAM bandwidth (§5.3).
- The planner keeps a 2 × 128 MiB safety margin of VRAM (`migration_plan.rs:176`), and `set_device` reserves 768 MiB for a new context and 256 MiB for runtime internals (`constant.rs`).

## 4. Scheduling: the essence (§6)

Problem: which app should hold the GPU, and for how long, with no user annotations (§6).

Idea: **MLFQ from CPU scheduling, with GPU-scale time slices and switching costs.** An app that uses its whole slice is treated as batch and demoted; an app that goes idle quickly (a chat or code-completion request that finishes) is treated as interactive and promoted. Higher level runs first. Same level runs round-robin, longest-waiting first (§6.3).

Differences from CPU MLFQ, and why:

- **Slices are seconds, not milliseconds**, because a switch moves gigabytes. Paper: demotion allotment T = 8 s and preemption threshold S = 4 s at the top level, doubled per level (§6.3). Code: demotion quantum 8/16/32/64/128 s; after every switch a **cooldown** = max(config, 3 × estimated migration time at 16 GB/s, the level's quantum) must pass before an equal-priority app may preempt (`policy.rs:74-160`). So in code the effective top-level slice is 8 s.
- **No periodic priority reset**, because a reset throws away seconds-long history and suddenly preempts interactive work (§6.2). The code has the reset commented out.
- **Soft promotion (Alg. 1).** An idle app is promoted if it stayed idle long enough, where time spent *waiting in the queue* mostly does not count as idle. Otherwise an app that is starved in the queue would look idle and get promoted, preempting the running app and shrinking everyone's slice (§6.2). Paper: promote if `i_a − R·q_a > T_{p−1} + t_a`, with `R < 1/N`. Code (`policy.rs:428-471`): promote if `idle − q·(m−1)/m > T/2 + t_a` with `m = max(N, 2) + 1`, and only if the priority has not changed for 2 T.
- **Preemption only when someone asks.** The running app is never stopped on a timer alone. A switch happens when another app blocks in the gate and sends `RequestScheduling`, and the queue head has priority ≥ the running app (equal priority: after the cooldown) (`policy.rs:239-303, 524-556`).
- **Dynamic floor.** Automatic demotion stops at `Batch`; `Background` is only reachable by `nixie priority set` (`policy.rs:420`).
- **Prefetch from the queue.** If the head of the queue cannot run yet, its data is moved up the host tiers now, so its later switch only has to cross PCIe (§6.3). Worth about 5% throughput in the batch case (Fig. 14).
- **Queue order**: pending idle notices first, then prefetch requests, then scheduling requests. Yield requests (apps that hit OOM and hold the metadata lock) sort before others (`policy.rs:239-303, 489-521`).

```
levels (code)      demote after   dynamic?
Interactive   4    8 s            start here
LowInteractive 3   16 s
HighBatch     2    32 s
Batch         1    64 s           floor for dynamic
Background    0    128 s          manual only
```

## 5. How a library gets "hotness" (it mostly does not) (§6.1)

Nixie has **no memory hotness information at all**: no access counters, no fault sampling, no LRU of chunks. The only signal it collects is **how recently the app called CUDA**, per API class, from timestamps taken inside the hooks. That tells it whether an app is *active* or *idle*, which drives both the scheduler and "lazy" residency.

Why not NVML utilisation: it takes about 600 ms to show that a workload stopped, and background desktop GPU work adds noise (§6.1).

Where the timestamps are taken (`sidecar/src/schedule/stats.rs`, set from the hooks in part 1.2):

```
cudaLaunchKernel     → last_kernel
cudaGraphLaunch      → last_graph
cudaMalloc*          → last_malloc
cudaMemcpy*/Memset   → last_transfer, last_transfer_size
cudaMemcpy (H2D/D2H) → blocking_transfer_start / _end
cuda*Synchronize     → sync_start / sync_end
```

A monitor thread in each app checks every 20 ms (`sidecar/src/schedule/mod.rs:219-285`, trimmed):

```rust
if state == Running && since_running > 100ms                       // anti-flap cooldown
   && no sync in progress && no blocking copy in progress           // blocked ≠ idle (§6.1)
   && kernel_elapsed   > 100ms && graph_elapsed > 200ms
   && malloc_elapsed   > 300ms
   && (transfer_elapsed > 300ms || last transfer was small)
   && blocking_transfer_elapsed > 100ms && sync_elapsed > 100ms
{
    state = Paused;                       // stop accepting launches without a re-request
    send(ActivityUpdate::Idle);           // daemon: make_resident_idle(), LastActive
}
```

The paper states a single 100 ms threshold (§6.1); the code uses 100–300 ms by API class. Blocking calls are bracketed start and end, so an app waiting inside `cudaStreamSynchronize` for a long kernel is not called idle (§6.1).

The daemon keeps per-app statistics (`daemon/src/runtime/schedule/statistics.rs`): time used at the current level (`t_a`), time since last priority change (`p_a`), idle since (`i_a`), time in the schedule queue (`q_a`), and a history of run slices with stop reasons (`nixie history --pid`).

What this means for placement: an idle app's data is not "cold data", it is "an app nobody is using right now". Inside an app, every chunk is treated alike: all of it comes in when the app runs, and victims are picked by size, not by use.

## 6. Results worth remembering (§7)

Testbed: Ryzen 9 9950X, 96 GB DDR5, 2 × RTX 5090 32 GB on PCIe 5.0 x8, CUDA 12.9, driver 580.95 (§7). Second testbed: 8 × RTX A5000 24 GB, PCIe 4.0 x16, driver 550.67 (§7.3).

- Context switch (time to first token): 44–82% lower than UVM/nvshare [3] for Ollama models, 30–36% lower for SGLang (Fig. 6).
- Transfer: about 2× UVM, near `nvbandwidth` [22] (Fig. 7).
- Pinned memory: same latency as UVM with 33–40% of UVM's pinned memory (Fig. 9).
- Launch overhead: none measurable on ResNet at batch 1 (Fig. 10a). `cudaMalloc` slightly slower between 2 and 128 MiB, faster above (Fig. 10b).
- Code completion next to a long-running agent: 3.1–3.8× faster than nvshare with W = 4 s (Case #3, Fig. 13).
- Batch fairness: about 85% of ideal throughput, similar to nvshare W = 30 s (Case #4, Fig. 14).

Baseline caveat: their "UVM" baseline hooks only `cudaMalloc`, `cudaFree`, `cudaMemGetInfo` (no advise, no prefetch) (§7). TGS [34] was only used in case studies.

## 7. Paper vs. code (eebf583)

| Topic | Paper | Code |
|---|---|---|
| Idle threshold | 100 ms (§6.1) | 100–300 ms per API class |
| Top-level preemption S | 4 s (§6.3) | cooldown ≥ 8 s quantum; a 4 s table exists but is unused |
| Promotion (Alg. 1) | `i_a − R·q_a > T_{p−1} + t_a`, `R < 1/N` | `idle − q·(m−1)/m > T/2 + t_a`, `m = max(N,2)+1` |
| Hooked APIs | includes `cudaStreamCreate` (§4) | not hooked; fixed 256 MiB / 768 MiB reservations |
| Number of levels K | unspecified | 5 |
| Code size | ~10k lines Rust (§7) | ~13.3k lines |
| VA release | not discussed | `cuMemAddressFree` never called |

## 8. What Nixie does not do, and what it means for libvmem

Limits the paper states (§8): temporal multiplexing only, no spatial sharing; no tensor semantics, so immutable weights are copied out even when a host copy exists ("white-box" hints named as future work); single-user threat model.

Limits from the code: whole-app granularity (all of an app's memory in, or it does not run); victim choice by size and list order, no hotness; runtime-API-only interposition (part 1.3).

Relevance to our design (this project's reading, not the paper's): Nixie already shows VMM remapping behind stable pointers, a pinned staging tier with exactly-one-copy, and full-duplex pipelined migration in userspace. It gives no per-buffer control: no hints, priorities, or streaming inside one app. `libvmem` differs by sharing spatially, with a pinned host fallback and optional per-buffer hints (tracker goal 1).

## References cited above (from the paper's bibliography)

[1] CUDA Unified Memory · [3] Alexopoulos, Mitropoulos, *nvshare*, ICSE-Companion '24 (repo <https://github.com/grgalex/nvshare>, confirmed via `gh`) · [4] Allen, Ge, *Demystifying GPU UVM cost*, IPDPS 2021 · [9] ComfyUI · [13] llama.cpp · [21] llama-swap · [22] NVIDIA nvbandwidth · [23] Unified Memory on Windows, CUDA Programming Guide 13.1 · [24] CUDA Driver API, Virtual Memory Management · [25] Ollama · [34] Wu et al., *TGS*, NSDI '23 · [36] Xiang et al., *Aegaeon*, SOSP '25 · [39] Yu et al., *Prism*, 2025 · [42] Zheng et al., *SGLang*, NeurIPS 2024. Related work (§9): PipeSwitch [6], DeepUM [16], G10 [40], XSched [28], Autellix [20], Nexus [27], Shepherd [41], Clockwork [14], ServerlessLLM [11].
