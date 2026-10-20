# vmemd

`vmemd` is the daemon that lets several processes share one GPU through libvmem (tracker 1Bii). The broker is `daemon/vmemd.cpp` and test 4 checks it. The client is the `vmem` backend, `src/backends/vmem.h`, and test 5 checks it.

## Layers

We split the work into three layers. Each layer sees less than the one above it.

| Layer | Code | Sees | Does |
|---|---|---|---|
| Allocator | `src/libvmem.cuh` | objects (the pointers the app holds) | packs objects into chunks, synchronizes before moves and frees, public API |
| Backend | `src/backends/*.h` | chunks (multiples of 2 MiB) | creates, maps, moves and destroys chunks inside the app's process |
| Broker | `daemon/vmemd.cpp` | chunk sizes and locations, per tenant | picks device or host for each new chunk, approves moves, keeps accounts |

We call each process connected to the broker a tenant. Only the `vmem` backend talks to the broker. The `cuda` and `uvm` backends never do.

The broker holds no memory. The backend calls `cuMemCreate` and maps each chunk in its own process, and the broker only says where to put it. No handles cross process boundaries, so a crashed tenant needs no cleanup: the driver frees its memory, its socket closes, and the broker drops its accounts.

## Messages

Backend and broker talk over a Unix socket (`SOCK_SEQPACKET`, default `/tmp/vmemd.sock`, set with `Config::daemon` or `VMEM_DAEMON`). One epoll thread serves every tenant. The broker reads each tenant's pid from `SO_PEERCRED`.

We name a chunk by its pointer. The backend reserves the address range before it contacts the broker, so the name exists when the first message goes out.

| Message | Fields | Reply |
|---|---|---|
| `Hello` | name | `Ok` |
| `Alloc` | ptr, size | `Ok` and a location, or `NoMemory` |
| `AllocFail` | ptr | none |
| `Free` | ptr | none |
| `Move` | ptr, target | `Ok` or `NoMemory` |
| `MoveDone` | ptr, where the chunk is now | none |
| `Resync` | ptr, size, location | none |

The name in `Hello` comes from `Config::name` and defaults to the program name. It only appears in the broker's log.

## Core Logic

The backend runs in one of two modes. In daemon mode the broker decides placement and approves moves. In local mode the backend decides alone. `link()` tells which one is active: `Connected`, `Local` (no broker was running at `init`), or `Broken` (the broker was lost later). A `Broken` link behaves like `Local` until the app calls `reconnect()`.

We describe the backend through what each call needs from the caller (pre), what it leaves behind (post), and what always holds (invariants). The allocator above it relies only on these.

### Backend invariants

- **B1.** Every live chunk is one address range, mapped read/write on the device over its whole size by exactly one memory handle. The address is fixed from `create` to `destroy`.
- **B2.** A chunk's `loc` is where its memory is. `used(Loc)` is the sum of the sizes of the live chunks at that location.
- **B3.** A call that fails leaves everything as it was before the call, with one exception: `Fatal` means the allocator must treat the context as dead.
- **B4.** No call blocks longer than 2 s on the broker, and none raises `SIGPIPE`. A failed send or receive turns the link `Broken` and the call carries on locally.
- **B5.** While the link is `Connected`, the broker's table for this tenant matches the backend's chunks, apart from messages without a reply that it has not processed yet.

### Calls

**`open`**
- Pre: none.
- Post `Ok`: the context exists, the granularity is the larger of the device and host granularity (so a move keeps the size), and `link()` is `Connected` if a broker answered `Hello`, else `Local`.
- A missing socket file or a refused connection means local mode. Any other connect error returns `Disconnected`.

**`create(size)`**
- Pre: `size` is a positive multiple of the granularity.
- Post `Ok`: the chunk is mapped (B1) at `out.loc`, and in daemon mode the broker counts it there.
- Post `NoMemory`: nothing was created and nothing is counted, either locally or at the broker (B3). The broker's refusal gives `NoMemory` even when CUDA still has room.
- Steps: reserve the address range, whose address names the chunk. In daemon mode send `Alloc` and take the location from the reply. Otherwise use the device. Create the memory there and map it. If the device create fails, create on the host and send `AllocFail`, so the broker moves the bytes to host in its counts. If anything still fails, free the range and, if the broker counted the chunk, send `Free`.

**`move(chunk, target)`**
- Pre: the chunk is live and idle, because the allocator has synchronized.
- Post `Ok`: same address, same data, `loc == target`.
- Post `NoMemory`: the chunk is unchanged (B3). The broker may have refused, or CUDA has no room at the target.
- Post `Fatal`: the chunk may have lost its backing. This happens only when the new memory will not map and the old memory cannot be mapped back, or when a CUDA call fails in a way the backend cannot undo.
- Steps: in daemon mode send `Move` and stop on `NoMemory`. Create memory at the target. Copy the old memory into it through a scratch address range and synchronize. Unmap the old memory, map the new one at the chunk's address, and release the old one. If the new memory will not map, map the old one back. In daemon mode send `MoveDone` with where the chunk ended up: the target on success, the old location on failure.

**`destroy(chunk)`**
- Pre: the chunk is live and idle.
- Post `Ok`: the memory and the address range are gone, `used` has dropped, and in daemon mode `Free` has been sent. The pointer is invalid.

**`reconnect`**
- Pre: none.
- Post `Ok`: the link is `Connected`, and the broker holds one entry per live chunk, including chunks created while the link was `Broken` (B5).
- Post `Disconnected`: the link is `Broken` and nothing else changed.
- Steps: open a new connection, send `Hello`, then send one `Resync` per live chunk. The backend never reconnects on its own.

### Broker invariants

- **K1.** For each tenant, the device count is the context overhead plus the bytes of its chunks on the device. The host count is the bytes of its chunks on the host. A granted move also holds its target bytes until `MoveDone`, `Free` or disconnect. A tenant has at most one move in flight.
- **K2.** The global counts are the sums over tenants. A disconnect removes the tenant's share.
- **K3.** `Alloc` and `Move` are the only requests the broker can refuse. It refuses when the target would pass its VRAM or DRAM limit, and it answers `Alloc` with the device while the device has room, then the host, then `NoMemory`.
- **K4.** `AllocFail`, `Resync` and `MoveDone` report facts, so the broker records them even when they put usage over a limit, and logs it. It never makes a tenant evict.

On a tenant's first `Hello`, the broker adds `ctx_overhead_mib` to its device count, because every CUDA context takes VRAM that libvmem never sees. A local probe measured a bare context at 90 MiB, 92 MiB after the first allocation, and 102 MiB with cuBLAS under lazy module loading. Eager loading raised the cuBLAS figure to 306 MiB. Under MPS each client took 81 MiB, and the MPS server took 37 MiB once. Kernels load lazily, so a running app takes more over time. The configured value is a rough estimate.

## Edge cases

- **`AllocFail` when the host is full.** The tenant already holds the host memory, so the broker accepts the chunk even over the DRAM limit and logs it.
- **A `Resync` that disagrees with the broker.** The broker trusts the tenant, since the tenant holds the memory. A pointer already in the table is overwritten. Usage over a limit is accepted and logged.
- **An unknown pointer.** A `Free`, `Move` or `AllocFail` for a pointer not in the table is a client bug. The broker ignores it (a `Move` gets `NoMemory`) and logs it.
- **A `MoveDone` with an unexpected location.** The broker records the location the tenant reports and logs it.
- **A `Free` during a move.** The broker drops the move and releases its reservation.
- **Messages from different tenants.** Messages without a reply (`Free`, `AllocFail`, `MoveDone`, `Resync`) are ordered only on their own socket. Another tenant's `Alloc` can arrive before an earlier `Free` and see the old counts, so it may land on the host or get `NoMemory` although room is about to appear. A tenant's own next request always follows its earlier messages.
- **A tenant sends a message of the wrong size.** The broker drops that tenant.

## Configuration

The broker reads its limits from a file passed with `--config PATH`. It also takes `--socket PATH` and `--log PATH` (default `/tmp/vmemd.log`). `daemon/vmemd.toml` is the example for this machine. The file holds `key = value` lines, and `#` starts a comment.

```
vram_limit_mib = 5000    # device memory the broker hands out
dram_limit_mib = 8000    # pinned host memory the broker hands out
ctx_overhead_mib = 100   # VRAM each tenant's CUDA context takes outside libvmem
```

For the two limits, a missing key or -1 means no limit. For `ctx_overhead_mib`, it means no overhead. The broker refuses to start on a missing file, an unknown key, a value below -1, or text after the number. We chose a file over reading the capacity at startup because the right limit depends on what else runs on the machine, and because asking CUDA for the VRAM size would create a context in the broker, which itself takes VRAM.

## Logging

The broker appends to the log file, one line per event, so a run can be reconstructed:

- a tenant connects, says `Hello`, or disconnects;
- each `Alloc`, `Free` and granted `Move`, and each `AllocFail`;
- an `Alloc` that gets `NoMemory`, and a refused `Move`;
- usage going over a limit after a `Resync` or an `AllocFail`;
- any edge case above.

## Open points

- We have not measured the cost of one blocking round trip per chunk. For comparison, the VMM calls alone take 3 to 230 µs.
- Per-tenant MPS limits (tracker 2C) are left for later.
- The protocol has no version field. Client and broker are built from the same `src/protocol.h` on the same machine. We add a version if they ever ship separately.
