# libvmem

**Status: work in progress (tracker 1B).** libvmem is a per-process allocator. Apps include `src/libvmem.cuh` and call it directly. It gets its memory from one backend, which the app picks at `init`: `vmem` (VMM), `uvm`, or `cuda`. All three backends work. On `vmem`, `evict` and `prefetch` move memory between device and host while keeping the same address; the `vmemd` daemon is not written yet, so `vmem` runs in local mode.

## Layout

```
src/libvmem.cuh     the allocator: pools, synchronization, public API (header only)
src/backend.h       backend interface and shared types (Loc, Status, Link, Chunk)
src/backends/       one header per backend: cuda.h, uvm.h, vmem.h
src/protocol.h      vmemd wire messages, shared by the vmem backend and the daemon
tests/              testNb.cu mains (N = test, b = backend: c cuda, u uvm, v vmem); testN.cuh bodies; utils.cuh helpers (pattern kernels, check printing)
tests/data/         small base images (PGM) and make_images.py; test3 tiles them up to VRAM-sized images
daemon/             vmemd broker (design only so far: README.md)
bin/                build output (not tracked)
```

## Build and run

```
make test N=1c         # build and run tests/test1c.cu; also 1u, 2c, 2u
make daemon            # build bin/vmemd
```

The tests, what each one checks, and the stress test results are in [`tests/README.md`](tests/README.md). The daemon design is in [`daemon/README.md`](daemon/README.md).

## API

```
vmem::init({.backend = vmem::Kind::Cuda});   // once per process; VMEM_BACKEND=vmem|uvm|cuda overrides
float *p = vmem::alloc<float>(n);            // or vmem::alloc(bytes); nullptr when out of memory
vmem::free(p);                               // just a free: no caching, no hints
vmem::Status s = vmem::evict(p);             // hard hint: p is idle, move it to host now
s = vmem::prefetch(p);                       // hard hint: move it back to the device now
vmem::get().info();                          // backend, daemon link, bytes held, chunk counts
vmem::get().used();                          // bytes in live chunks (rounded to 2 MiB), every backend
vmem::get().used(vmem::Loc::Host);           // per location; throws Unsupported on uvm, which cannot tell
vmem::shutdown();                            // also runs at exit
```

## Hints

`alloc` and `free` do not take hints. Instead, we add two calls, `evict(p)` and `prefetch(p)`, which move one large allocation right away. `libvmem.cuh` first calls `cuCtxSynchronize`, then asks the backend to `move` the chunk. The address stays the same and the data is kept. By calling a hint, the app promises that `p` is idle. Small objects share chunks with each other, so they never move.

Hints block until the copy is done. We plan to add non-blocking hints (tracker 1Bv). The copy itself can run on its own stream, but swapping the old backing for the new one (`cuMemUnmap` + `cuMemMap`) is a host call that no event or stream can order. The app would therefore have to call `wait(p)` before it touches `p` again.

A hint never fails the program. It returns a `Status`, and for anything other than `Ok` the chunk stays where it was:

| Status | Meaning |
|---|---|
| `Ok` | the chunk is at the target (also when it already was) |
| `NoMemory` | no room at the target |
| `Unsupported` | small object, pointer not from libvmem, or the backend cannot move (`cuda`) |
| `Fatal` | the CUDA context is dead; later `alloc` and `free` throw |

Other errors work as follows. `alloc` returns `nullptr` when the backend runs out of memory. Misuse, such as freeing a foreign pointer or freeing twice, throws `vmem::Error`. A fatal CUDA error (a dead context) also throws, and so does every call after it.

## Pools

The allocator sends each request to one of two pools, depending on its size:

| | SmallPool | LargePool |
|---|---|---|
| Takes | at most 1 MiB | more than 1 MiB |
| Chunks | 2 MiB, shared | one per allocation, rounded up to 2 MiB |
| Inside a chunk | power-of-two objects of one size, at least 512 B | the whole allocation |
| On free | slot reused; empty chunk released | chunk released |
| Moves | never | `evict` and `prefetch` |

We took the thresholds from PyTorch's caching allocator. Unlike PyTorch, our large pool does not cache freed chunks.

## Backends

A backend only sees chunks. It can `create` a chunk of any multiple of its granularity (and decides itself whether it goes on the device or the host), `destroy` it, and `move` it. A move keeps the same address. Backends also implement `open`, `close`, `link`, `reconnect`, and `used`. They never synchronize; the allocator does that for them.

`move` gives the **strong guarantee**: it either finishes, or it fails and leaves the chunk exactly as it was. It reports failure through its `Status` and never throws. The one exception is `Fatal`. On `vmem`, a move copies into new memory first and swaps it in under the chunk's address only after the copy succeeds. If the swap fails, the backend maps the old memory back. If that also fails, `move` returns `Fatal`, the chunk has no memory behind it, and the allocator marks itself dead.

| Backend | Memory | Moves |
|---|---|---|
| `cuda` | `cudaMalloc`, device only | no |
| `uvm` | `cudaMallocManaged`; `move` is a blocking `cudaMemPrefetchAsync` | yes, but `Ok` means requested, not resident |
| `vmem` | VMM chunks on the device until `cuMemCreate` fails, then pinned host memory | yes, same address |

On `uvm` the driver decides where pages live. `where()` therefore reports the last location we asked for, and `used(Loc)` throws `Unsupported`.

## Rules

- libvmem is single-threaded.
- `libvmem.cuh` calls `cuCtxSynchronize` (on the current context) before every backend `move` and `destroy`, so memory never moves or disappears under a running kernel.
- When the app calls `evict(p)` or `prefetch(p)`, it promises that `p` is idle.
