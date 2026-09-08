# VMMRemapShared

Two extensions to `../VMMVector`, built with the CUDA driver VMM API:

1. **`remap()`** — move a chunk's physical backing between GPU VRAM and host
   DRAM (`HOST_NUMA`) **without changing its virtual address**.
2. **Multi-process subscribe** — one owner process creates a VMM segment; other
   processes attach to the *same physical memory* and allocate from it through
   shared metadata.

Both demos pass on this box (RTX 4050, CUDA 13.0, single NUMA node).

See `../pytorch-vmm-study/README.md` for how PyTorch's `ExpandableSegment` uses
the same primitives — this code is deliberately the same shape.

---

## Build & run

```
make            # -> remap_test, owner, subscriber
./run.sh        # builds, runs both demos, exits non-zero on failure
```

Individually:

```
./remap_test [n_elems] [chunk_MB]          # default 1048576 2
./owner [n_subs] [n_chunks] [chunk_MB] &   # default 2 4 2
./subscriber A &  ./subscriber B &
```

`run.sh` cleans `/dev/shm/vmm_share_state` and `/tmp/vmm_share.sock` first.

---

## Part 1 — remap (`vmm_remap.cuh`, `remap_test.cu`)

`VMMRemapVector<T>` is a trimmed copy of `VMMVector` (same
reserve-VA / create / map / set-access flow). Added:

| Member | Purpose |
|---|---|
| `std::vector<char> chunk_on_host_` | per-chunk location (0 = DEVICE, 1 = HOST_NUMA) |
| `props_for(bool host)` | a `CUmemAllocationProp` for the target location |
| `remap(size_t chunk_idx, bool to_host)` | the move |
| `chunk_on_host(i)`, `n_chunks()` | observability |

### How `remap` works

VA of the chunk = `base = d_ptr + chunk_idx * chunk_size_`. That address never
changes; only the physical handle behind it does.

1. **`cuMemCreate`** a new handle at the target location (`HOST_NUMA` id 0 /
   `DEVICE` id 0 — they coincide here).
2. **`cuMemAddressReserve`** a scratch VA range, **`cuMemMap`** the new handle
   there, **`cuMemSetAccess`** (device descriptor).
3. **`cuMemcpyDtoD(scratch, base, chunk_size_)`** + **`cuCtxSynchronize`** —
   copies live contents into the new backing. Works across locations because a
   `HOST_NUMA` allocation with a device access descriptor is reachable by the
   copy engine over the bus (this is the EGM model).
4. **`cuMemUnmap(base)`**, **`cuMemRelease`** the old handle,
   **`cuMemMap(base, …, new_handle)`**, **`cuMemSetAccess(base)`** — the swap.
5. **`cuMemUnmap(scratch)`** + **`cuMemAddressFree(scratch)`**. The new handle
   stays alive because it is still mapped at `base`.

Key point: **one device access descriptor covers both cases.** Kernels and
`cuMemcpy` reach host-backed chunks transparently; nothing above the allocator
needs to know a chunk moved.

### Test results (`./remap_test 1048576 2`, 2 chunks x 2 MB)

```
filled + doubled: 1048576 elems across 2 chunks
remapped 2 chunks device->host in ~1.4 ms, VA + contents intact
remapped 2 chunks host->device in ~0.9 ms, kernel ran, values = 4*i
=== remap_test PASSED ===
```

Asserts checked: `d_ptr` unchanged after every remap; contents survive
device→host (verified via `cuMemcpyDtoH`); a kernel runs correctly after
host→device. Latency is per-2-chunk, dominated by `cuMemCreate` +
`cuMemcpyDtoD` over the 4 MB.

---

## Part 2 — multi-process subscribe (`ipc_common.h`, `owner.cu`, `subscriber.cu`)

### Roles

- **owner** — creates the segment, owns all `cuMemCreate` handles, owns the
  metadata, is the barrier/coordinator.
- **subscriber** — imports the physical handles, maps them at *its own* virtual
  base, bump-allocates regions from the shared metadata, tags them on the GPU.

### Where shared state lives

**POSIX shared memory** (`shm_open("/vmm_share_state")` + `mmap`), a plain
struct (`ipc_common.h : SharedState`):

```c
pthread_mutex_t lock;      // PTHREAD_PROCESS_SHARED, initialised by the owner
uint64_t chunk_size, chunk_count, total_bytes, bump;   // bump = next free offset
int      n_allocs;
int      chunk_on_host[256];                            // observability
struct { uint64_t off,len; int pid; uint32_t tag; } allocs[64];
```

Allocations are **offsets, not pointers** — each process turns an offset into an
address with its own `base + off`. The demo forces the two subscribers to
different bases (owner `0x302000000`, subs `0x302200000` / `0x302a00000`) and
the same offsets still resolve to the same physical bytes in all three
processes.

### Wire protocol (UNIX domain socket, `/tmp/vmm_share.sock`)

Physical-memory handles cross the process boundary as **POSIX file descriptors**
(`cuMemExportToShareableHandle` → `sendmsg` with `SCM_RIGHTS` →
`cuMemImportFromShareableHandle`). Fixed-size `Msg` structs, fds in the ancillary
data:

| Msg | Direction | Payload |
|---|---|---|
| `MSG_HEADER` | owner → sub | `chunk_size`, `chunk_count`, + one fd per existing chunk |
| `MSG_ADD_CHUNK` | owner → sub | new `chunk_count` + 1 fd — sent when the segment grows |
| `MSG_GO` | owner → sub | barrier release |
| 1 byte `"D"` | sub → owner | "my alloc pass is done" |

Flow: connect → `HEADER` → sub maps N chunks, does pass-1 alloc+tag, sends `D` →
owner barriers on all `D`, creates chunk N, sends `ADD_CHUNK` to each → sub maps
it, does pass-2 alloc+tag in the grown region, sends `D` → owner barriers,
sends `GO` → everyone verifies every region through their own mapping.

### Demo output (`./run.sh`, owner + subs A, B)

```
[owner] segment: 4 chunks x 2 MB, base=0x302000000
[A 16034] mapped 4 chunks at own base 0x302a00000
[B 16035] mapped 4 chunks at own base 0x302200000
[owner] grew segment to 5 chunks, streamed to subscribers
=== owner: all 4 cross-process regions verified ===
[A 16034] verified 4/4 regions via own mapping — PASS
[B 16035] verified 4/4 regions via own mapping — PASS
```

Each of the 4 regions (2 subscribers × 2 passes, one pass in a dynamically
added chunk) is written by one process and read back correctly by all three
through three different virtual bases.

---

## What works / what's stubbed

Works:
- remap device↔host, VA-stable, contents preserved, kernel-safe after.
- cross-process handle sharing via `SCM_RIGHTS` fds.
- shared bump allocator under a process-shared mutex.
- dynamic segment growth streamed to live subscribers.

Stubbed / deliberately minimal (`ponytail:` comments in code):
- **Bump allocator, no free.** `bump` only goes up. Real use needs a free list.
- **One global lock** over the whole `SharedState`. Fine at this scale; a real
  design wants per-region or lock-free metadata.
- **Fixed tables** (64 allocs, 256 chunks), no overflow handling beyond an abort.
- **No length-prefixed framing** on the socket — relies on fixed `Msg` size and
  strict request/response alternation. A real protocol needs framing.
- **No teardown coordination.** Subscribers just exit; the owner unlinks shm and
  the socket. A subscriber that exits mid-run leaves its regions tagged but
  unreclaimed.
- remap is whole-chunk and synchronous (`cuCtxSynchronize` inside).

## Open questions (for the strawman)

- **Crash recovery of shared state.** If the owner dies, the shm mutex can be
  left locked (need `pthread_mutexattr_setrobust` + `EOWNERDEAD` handling) and
  the exported fds' backing is released when the owner's handles drop. Who
  owns the physical memory's lifetime — owner, or first/last subscriber?
- **Lock granularity.** A single mutex serialises every allocation across every
  tenant. Per-chunk freelists? A lock-free bump region per tenant with periodic
  compaction? Move allocation decisions out of the shared path entirely and
  have the owner serve them over the socket?
- **Remap while a subscriber holds a mapping.** If the owner remaps a chunk
  device→host, every subscriber's existing mapping of that chunk still points
  at the *old* physical handle (now released) → stale/faulting. Remap in a
  shared segment needs a revocation protocol: notify subscribers, have them
  `cuMemUnmap` + re-import the new handle, ack, then the owner completes. This
  is the hard part and is not implemented here.
- **Placement of the coordinator** — the shared-state question from the tracker:
  userspace daemon (this demo, minus the barrier hack) vs a kernel module vs a
  CUDA driver extension. This demo is the userspace-daemon baseline.

## References

- CUDA Driver API — Virtual Memory Management:
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html>
- NVIDIA blog, *Introducing Low-Level GPU Virtual Memory Management*:
  <https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/>
- CUDA Programming Guide, *Extended GPU Memory (EGM)*:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/egm.html>
- `../pytorch-vmm-study/README.md` — `ExpandableSegment::share()` / `fromShared()`
  do the same fd-passing (they use `pidfd_getfd`; this uses `SCM_RIGHTS`).
