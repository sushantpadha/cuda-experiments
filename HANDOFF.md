# HANDOFF

State of the VMM allocator R&D, for whoever picks it up next.

Last updated: 2026-09-09 · Puru / Teja / Sushant
Progress tracker: https://claude.ai/code/artifact/dcf94637-253f-4233-9124-d96bfa7c5c09

## Where things are

Goal: a VMM-based GPU allocator that can **remap backing store to host DRAM**
and be **shared across processes** for multi-tenant use.

Both halves now exist as working code in `VMMRemapShared/`:

| Piece | File | Status |
|---|---|---|
| `cudaremap` primitive (swap a VA's backing device↔host, VA fixed) | `remap.cuh` | works, round-trip tested |
| `cudaremap` on the growable vector | `vmm_remap.cuh` | works |
| single-process demo | `remap_test.cu` | passes |
| multi-tenant allocator (owner + tenants, shared segment) | `owner.cu`, `subscriber.cu`, `ipc_common.h` | works |
| live remap of a shared chunk (revoke/migrate/re-import) | in `owner.cu` / `subscriber.cu` | works, data survives, verified from 3 VA bases |

Study notes on how PyTorch does the same VMM dance: `pytorch-vmm-study/README.md`.
Base growable vector (untouched): `VMMVector/`.

## Run it

```
cd VMMRemapShared
./run.sh          # both demos -> example_output.txt (a committed passing run)
```

Needs `nvcc` (CUDA 13.0), `-lcuda -lpthread -lrt`, GPU with `HOST_NUMA`
(RTX 4050 cc 8.9 here). NUMA node id == device id == 0 on this box, hard-coded.

## Settled facts

- VMM is **pinned-only** — no `cudaMallocManaged` through `cuMemCreate`.
- VMM backing is placeable: `DEVICE` or `HOST_NUMA`. Both tested.
- Remap without VA change = unmap + re-back the same reserved range. Contents
  moved with a staged `cuMemcpyDtoD` through a scratch VA. ~1.3 ms dev→host,
  ~0.8 ms host→dev per 4 MB — copy-bound.
- IPC works with VMM (`cuMemExportToShareableHandle` → POSIX fd), not with
  managed memory.
- A device access descriptor reaches `HOST_NUMA` memory over the bus (EGM-style),
  so kernels/copies work on host-backed chunks unchanged.
- PyTorch / llama.cpp / vLLM all use VMM the same way (reserve big VA, map PA on
  demand) and **none** share the segment across processes.

## Design as built

- **Shared state**: POSIX shm (`shm_open`+`mmap`), one `SharedState` struct with
  a `PTHREAD_PROCESS_SHARED` mutex, a bump offset, a fixed alloc table, and a
  per-chunk location table. Allocations are **byte offsets**, valid in every
  process regardless of its virtual base.
- **Owner** is the single writer of segment geometry and the remap coordinator.
- **Remap handshake**: owner broadcasts `MSG_REVOKE{chunk}` → tenants unmap +
  release that chunk, ack → owner runs `remap_backing` (only it maps the chunk
  now) → owner re-exports, broadcasts `MSG_REMAP{chunk}` + new fd → tenants
  re-import + map at the same slot, ack → owner proceeds. Not transparent: a
  tenant gets an explicit revoke callback.

## Open / next (roughly ordered)

1. **`boxd`** — still not reviewed. Carried from week 1.
2. **Real free list** in the shared allocator (currently bump, no free).
3. **Eviction policy** — who decides a chunk moves to host, on what signal
   (memory pressure, idle time, explicit tenant hint)?
4. **Crash recovery** — owner or tenant crash leaves the shm mutex and segment
   undefined. Needs `pthread_mutexattr_setrobust`, owner liveness check, tenant
   re-attach path.
5. **Handshake robustness** — currently blocks on every tenant acking; a dead
   tenant stalls it. Needs timeout + forced eviction.
6. **Coordinator placement** — owner is a SPOF. Options weighed on the tracker:
   userspace daemon / kernel module / driver. Current pick: userspace, revisit.
7. **Hide the revoke window** per tenant with a fault handler
   (`cuMemSetAccess` PROT_NONE + handler that blocks until re-mapped) instead of
   the explicit callback — the "transparent" version, deferred on purpose.

## Repo map

```
VMMVector/          base growable GPU vector on the VMM API (unmodified)
VMMRemapShared/     cudaremap primitive + multi-tenant allocator  <- active work
pytorch-vmm-study/  notes: VMM usage in PyTorch's CUDACachingAllocator
transpose/ vecAdd/  unrelated small kernels
props.cu test.cu    device query / probes
```

`.gitignore` is allowlist-style (add extensions to track them). Binaries are
untracked; `*.txt` are captured run output.
