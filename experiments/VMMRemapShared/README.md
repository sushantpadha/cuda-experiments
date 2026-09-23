# VMMRemapShared

Two things built on the CUDA low-level VMM driver API:

1. **`cudaremap` primitive** — move a virtual range's physical backing between
   GPU VRAM and host DRAM (`HOST_NUMA`) **without changing the virtual address**,
   contents preserved. Plain VMM has no such call.
2. **Multi-tenant allocator** — one owner process holds a shared VMM segment;
   other processes attach, map the same physical memory at their own virtual
   base, and allocate from a shared offset-based bump allocator. The owner can
   `cudaremap` a chunk while tenants are attached, via a revoke/remap handshake.

Nothing in `../VMMVector/` is modified. `remap()` is written so it drops
straight back into the original `VMMVector`.

Study notes on how PyTorch drives these same APIs: [`../pytorch-vmm-study/README.md`](../pytorch-vmm-study/README.md).

---

## Files

| File | What |
|---|---|
| `remap.cuh` | `remap_backing(va, size, handle&, to_host, acc, handle_types)` — the primitive. `remap_granularity()` helper. |
| `vmm_remap.cuh` | `VMMRemapVector<T>` — the growable vector from `../VMMVector/vmm_vector.cuh` with `remap(chunk_idx, to_host)`. |
| `remap_test.cu` | Demo 1: single process. Fill on device → remap all chunks to host → verify VA + contents → remap back → kernel → verify. |
| `ipc_common.h` | `SharedState` (shm layout + process-shared mutex + bump allocator + chunk-location table), `SCM_RIGHTS` fd send/recv, wire `Msg`. |
| `owner.cu` | Demo 2 owner: creates chunks, serves fds, coordinates grow + remap. |
| `subscriber.cu` | Demo 2 tenant: imports chunks at its own base, bump-allocates, serves the remap handshake, verifies. |
| `run.sh` | Builds, runs both demos, tees `example_output.txt`. |
| `example_output.txt` | Captured passing run (RTX 4050, CUDA 13.0). |

## Build & run

```
make            # or: make debug   (-g -G -DDEBUG)
./run.sh        # both demos -> example_output.txt
./remap_test [n_elems] [chunk_MB]
./owner [n_subscribers] [n_chunks] [chunk_MB]   # subscribers: ./subscriber <label>
```

Needs `nvcc` (CUDA 13.0), `-lcuda -lpthread -lrt`, a GPU with `HOST_NUMA`
support (RTX 4050, cc 8.9 here). NUMA node id and device id are both `0` on this
box — hard-coded; change `location.id` elsewhere.

---

## How `cudaremap` works

`remap_backing(va, size, h, to_host, acc, handle_types)`:

1. `cuMemCreate` a new handle at the target location (`HOST_NUMA` or `DEVICE`).
2. `cuMemAddressReserve` a scratch VA, `cuMemMap` the new handle there,
   `cuMemSetAccess`, `cuMemcpyDtoD` the live bytes `va → scratch`
   (a *device* access descriptor reaches host memory over the bus — EGM-style),
   `cuCtxSynchronize`.
3. `cuMemUnmap(va)` + `cuMemRelease(old handle)`, then `cuMemMap(va, new handle)`
   + `cuMemSetAccess` — the virtual address is now backed by the new location.
4. `cuMemUnmap` + `cuMemAddressFree` the scratch VA. The new handle stays alive
   (mapped at `va`).

Cost measured here: ~1.3 ms device→host, ~0.8 ms host→device for 4 MB (2 chunks),
dominated by the staged copy.

### Multi-process remap (owner + tenants)

A tenant mapping the same physical handle keeps the old backing alive and would
see stale device memory after a migration. So the owner runs a handshake:

```
owner: broadcast MSG_REVOKE{chunk}          tenants: cuMemUnmap + cuMemRelease that chunk, ack
owner: remap_backing(owner_va, ...)          (only the owner maps the chunk now — clean copy)
owner: cuMemExportToShareableHandle(new)
owner: broadcast MSG_REMAP{chunk} + new fd   tenants: cuMemImportFromShareableHandle + cuMemMap
                                                      at the SAME slot, cuMemSetAccess, ack
owner: proceed
```

Tenant VAs never move; only that chunk's backing does. Regions in other chunks
are untouched. This is **not transparent** — a tenant gets a revoke callback and
its mapping of that chunk is briefly gone — which matches the current goal.

---

## Where shared state lives

POSIX shared memory (`shm_open` + `mmap`), one `SharedState` struct:

- `pthread_mutex_t` with `PTHREAD_PROCESS_SHARED` — owner inits it in the shm.
- bump offset + fixed `allocs[]` table — allocations are **byte offsets** into
  the segment, valid in every process regardless of its virtual base.
- `chunk_on_host[]` — location of each chunk, for observability.

Owner is the single writer of segment geometry (`chunk_count`, `total_bytes`);
tenants only bump-allocate under the lock.

---

## What works / what's stubbed

Works: primitive + round trip, cross-process map at distinct bases, segment
growth streamed to tenants, live device↔host remap of a chunk with tenants
attached, data survival verified from 3 address spaces.

Stubbed (`ponytail:` comments in the source):

- bump allocator, no free — real free list needed for anything long-lived
- one global lock over `SharedState` — per-region locks if contended
- fixed `MAX_CHUNKS` / `MAX_ALLOCS` tables
- wire protocol is fixed-size `Msg` + strict request/ack alternation, no framing
- **no crash recovery** — an owner or tenant crash leaves the shm lock and the
  segment in an undefined state; needs robust mutexes (`pthread_mutexattr_setrobust`)
  and an owner liveness check
- remap handshake blocks on every tenant acking; a slow/dead tenant stalls it —
  needs a timeout + eviction path

## Open questions for the strawman

- Owner as coordinator is a single point of failure. Move geometry + lock into a
  small daemon, or the driver, or make ownership transferable?
- Remap policy: who decides a chunk should be evicted, on what signal
  (pressure, idle time, an explicit tenant hint)?
- Can the revoke window be hidden per-tenant with a fault handler
  (`cuMemSetAccess` PROT_NONE + a handler that blocks until re-mapped) instead of
  an explicit callback?

## References

- CUDA Driver API — Virtual Memory Management:
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html>
- CUDA Programming Guide — Extended GPU Memory (EGM):
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/egm.html>
- NVIDIA blog — Introducing Low-Level GPU Virtual Memory Management:
  <https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/>
