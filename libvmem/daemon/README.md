# vmemd

This is the design for `vmemd`, the daemon that lets several processes share one GPU through libvmem (tracker 1Bi and 1Bii). The broker side (`daemon/vmemd.cpp`) handles every message and is tested by test 4. The client side in the `vmem` backend and the context overhead on `Hello` are not written yet.

We split the work into three layers. Each layer sees less than the one above it:

| Layer | Code | Sees | Does |
|---|---|---|---|
| Allocator | `src/libvmem.cuh` | objects (the pointers the app holds) | packs objects into chunks, synchronizes before moves and frees, public API |
| Backend | `src/backends/*.h` | chunks (multiples of 2 MiB) | creates, maps, moves and destroys chunks inside the app's process |
| Broker | `daemon/vmemd.cpp` | chunk sizes and locations, per tenant | picks device or host for each new chunk, approves moves, keeps accounts |

We call each process connected to the broker a tenant. Only the `vmem` backend talks to the broker; the `cuda` and `uvm` backends never do.

The broker does not hold any memory itself. The backend calls `cuMemCreate` and maps each chunk in its own process, and the broker only tells it where to put the chunk. Since no handles cross process boundaries, a crashed tenant needs no special cleanup: the driver frees its memory, its socket closes, and the broker drops its accounts.

## Messages

The backend and the broker talk over a Unix socket (`SOCK_SEQPACKET`, by default `/tmp/vmemd.sock`). The broker serves every tenant from a single epoll thread. It learns each tenant's pid from `SO_PEERCRED`, so tenants do not send it.

We name a chunk by its pointer. The backend reserves the address range before it asks the broker for memory, so the pointer already exists when the first message goes out.

| Message | Fields | Reply |
|---|---|---|
| `Hello` | optional name | `Ok` |
| `Alloc` | ptr, size | `Ok` and a location, or `NoMemory` |
| `AllocFail` | ptr | none |
| `Free` | ptr | none |
| `Move` | ptr, target | `Ok` or `NoMemory` |
| `MoveDone` | ptr, where the chunk is now | none |
| `Resync` | ptr, size, location | none |

When the allocator needs a new chunk:

1. The backend reserves the address range and sends `Alloc`.
2. The broker picks a location, adds the chunk to the tenant's usage there, and replies with that location.
3. The backend creates the memory at that location and maps it.

The device `cuMemCreate` in step 3 can still fail, because the broker cannot see everything that uses VRAM: the CUDA context, plain `cudaMalloc` calls, or processes outside libvmem. In that case the backend falls back to host memory and sends `AllocFail`. The broker then moves the chunk's bytes from device to host in its counts and logs the event.

When the app calls `evict` or `prefetch`:

1. The allocator calls `cuCtxSynchronize`, so no kernel is using the chunk.
2. The backend sends `Move`. The broker either reserves room at the target and answers `Ok`, or answers `NoMemory`, in which case the chunk stays where it is.
3. The backend copies the data, swaps the old backing for the new one, and sends `MoveDone` with the chunk's final location. That is the target if the move worked and the old location if it failed. The broker releases the reservation and updates its accounts.

When the allocator frees a chunk, the backend destroys it and then sends `Free`. The broker does not reply.

## Accounts

For each tenant, the broker keeps:

- its pid, optional name, socket, and connect time;
- a chunk table that maps each pointer to its size and location;
- the move in flight, if any;
- device bytes (its chunks plus its context overhead) and host bytes;
- its VRAM and DRAM limits, which stay unlimited until we add quotas (1Bvi);
- a priority, which v1 does not use (see 1Bviii).

Across all tenants, it keeps the VRAM and DRAM capacity, the VRAM limit (capacity minus a margin), the DRAM limit, the per-context overhead, and running device and host totals.

## Configuration

The broker reads its limits from a config file, which you must pass with `--config PATH`. `daemon/vmemd.toml` is an example for this machine. The file holds `key = value` lines, and `#` starts a comment:

```
vram_limit_mib = 5000   # what the broker hands out on the device
dram_limit_mib = 8000   # and in pinned host memory
```

A key that is missing or set to -1 means no limit. The broker refuses to start on an unknown key, a value below -1, or anything after the number, and it logs both limits when it starts. We chose a file over reading the capacity at startup because the right limit depends on what else runs on the machine, and because asking CUDA for the VRAM size would create a CUDA context in the broker, which itself takes VRAM.

## Placement

The broker places new chunks on the device until the VRAM limit is reached, then on the host until the DRAM limit is reached. After that, `Alloc` gets `NoMemory`. When a tenant frees device memory, the next request can use it.

Every CUDA context takes some VRAM that libvmem never sees. To account for it, the broker adds one context's overhead to a tenant's device usage when the tenant sends `Hello`.

Moves follow the same limits: an `evict` needs room under the DRAM limit, and a `prefetch` needs room under the VRAM limit.

## Failures

- **A tenant crashes.** The driver frees its memory and its socket closes, so the broker just drops its accounts.
- **No broker is running at `init`.** The backend runs in local mode: it puts chunks on the device until `cuMemCreate` fails, then on the host.
- **The broker restarts.** `reconnect()` sends `Hello` and then one `Resync` for each live chunk. The broker accepts the reported chunks as they are, even if that puts usage over the limit, and logs it. It then sends new chunks to the host until usage drops below the limit. It never forces a tenant to evict.

## Edge cases

- **`AllocFail` when the host is already full.** By the time the broker hears about it, the tenant already holds the host memory. The broker accepts the chunk even though it puts host usage over the DRAM limit, the same way it treats `Resync`.
- **A `Resync` that does not match the broker's view.** The broker trusts the tenant's report, since the tenant holds the memory. If a pointer is already in the tenant's chunk table, the broker overwrites the entry. If the reported chunks put usage over a limit, it accepts them and logs it.
- **An unknown pointer.** A `Free`, `Move`, or `AllocFail` for a pointer missing from the tenant's chunk table is a client bug. The broker ignores the message (a `Move` gets `NoMemory`) and logs it.
- **A `MoveDone` with an unexpected location.** If the reported location is neither the target nor the old location, the broker records what the tenant reported and logs it.
- **Messages from different tenants.** Messages without a reply (`Free`, `AllocFail`, `MoveDone`, `Resync`) are ordered only on their own socket. Another tenant's `Alloc` can reach the broker before an earlier `Free`, and then sees the old counts: it may land on host or get `NoMemory` although the room is about to be freed. A tenant's own next request always comes after its earlier messages.

We do not version the protocol. The client and the broker are built from the same `src/protocol.h` on the same machine; we will add a version field if they ever ship separately.

## Logging

The broker writes one info line for each of these events, so a run can be reconstructed from the log:

- a tenant connects or disconnects, with its pid, name, connect time, and its device and host usage at that moment;
- an `AllocFail`, with the chunk size;
- usage goes over a limit after a `Resync` or an `AllocFail`, with the limit and the new total;
- an `Alloc` that gets `NoMemory`, and a `Move` that is refused;
- any of the edge cases above.

## Open points

- We have not measured the per-context overhead yet.
- We have not decided what the backend does between losing the broker and calling `reconnect()`.
- We have not measured the cost of one blocking round trip per chunk. For comparison, the VMM calls alone take 3 to 230 µs.
- Per-tenant MPS limits (tracker 2C) are left for later.
