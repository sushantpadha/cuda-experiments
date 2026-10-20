# libvmem

**Status: work in progress (tracker 1B).** libvmem is a per-process allocator. Apps include `src/libvmem.cuh` and call it directly. It gets its memory from one backend, which the app picks at `init`: `vmem` (VMM), `uvm`, or `cuda`. All three backends work. On `vmem`, `evict` and `prefetch` move memory between device and host while keeping the same address. When the `vmemd` daemon runs, the `vmem` backend asks it where each new chunk goes and whether a move may happen; without it, the backend runs in local mode. If the daemon is lost, the backend carries on in local mode until the app calls `reconnect()`, or, with `auto_reconnect` set, until the next `create` or `move` reconnects by itself.

## Layout

```
src/libvmem.cuh     the allocator: pools, synchronization, public API (header only)
src/backend.h       backend interface and shared types (Loc, Status, Link, Chunk)
src/backends/       one header per backend: cuda.h, uvm.h, vmem.h
src/protocol.h      vmemd wire messages, shared by the vmem backend and the daemon
tests/              test1.cu, test2.cu (first argument = backend: c cuda, u uvm, v vmem, d vmem with vmemd); utils.cuh helpers;
                    workload.cuh (churn, gemm, image: the work of bin/worker)
tests/daemon/       tests driven by Python checkers: state_*, pair_*, worker.cu + workers_check.py (stress and workloads); harness.py shared
tests/data/         small base images (PGM) and make_images.py; the image workload tiles them up to big images
daemon/             vmemd broker: vmemd.cpp, example config vmemd.toml, design in README.md
analysis/           memory traces (CSV) and plots from make trace-all
bin/                build output (not tracked)
```

## Build and run

```
make test N=1c         # bin/test1 c; also 1u, 1v, 1d, 2c, 2u, 2v, 2d (d = vmem with vmemd); make test-unit runs all eight
make stress            # four processes of random calls against one vmemd
make test-daemon       # test-state (one client), test-pair (two at once), stress, test-workers-gemm, -image and -mix (four processes each)
make daemon            # build bin/vmemd
```

The tests and what each one checks are in [`tests/README.md`](tests/README.md), and their measured results in [`tests/RESULTS.md`](tests/RESULTS.md). The daemon design is in [`daemon/README.md`](daemon/README.md).

## API

```
vmem::init({.backend = vmem::Kind::Cuda});   // once per process; VMEM_BACKEND=vmem|uvm|cuda overrides
                                             // .daemon = vmemd socket (default /tmp/vmemd.sock); VMEM_DAEMON overrides
                                             // .name = tenant name in the vmemd log (default: the program name)
                                             // .auto_reconnect = retry vmemd at each create/move while unlinked (default false: call reconnect())
float *p = vmem::alloc<float>(n);            // or vmem::alloc(bytes); nullptr when out of memory
vmem::free(p);                               // just a free: no caching, no hints
vmem::Status s = vmem::evict(p);             // hard hint: p is idle, move it to host now
s = vmem::prefetch(p);                       // hard hint: move it back to the device now
vmem::get().info();                          // backend, daemon link, bytes held, chunk counts
vmem::get().reconnect();                     // vmem: connect to vmemd again and report every live chunk
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

`move` gives the **strong guarantee**: it either finishes, or it fails and leaves the chunk exactly as it was. It reports failure through its `Status` and never throws. The one exception is `Fatal`. On `vmem`, a move copies into new memory first and swaps it in under the chunk's address only after the copy succeeds. If the swap fails, the backend maps the old memory back. If that also fails, `move` returns `Fatal`, the chunk has no memory behind it, and the allocator marks itself dead. A CUDA call that fails for any other reason than running out of memory (the copy, the unmap) also returns `Fatal`, and the allocator treats that as a dead context too.

| Backend | Memory | Moves |
|---|---|---|
| `cuda` | `cudaMalloc`, device only | no |
| `uvm` | `cudaMallocManaged`; `move` is a blocking `cudaMemPrefetchAsync` | yes, but `Ok` means requested, not resident |
| `vmem` | VMM chunks; `vmemd` picks device or pinned host memory, or, in local mode, the device until `cuMemCreate` fails | yes, same address; `vmemd` must grant it |

On `uvm` the driver decides where pages live. `where()` therefore reports the last location we asked for, and `used(Loc)` throws `Unsupported`.

## Rules

- libvmem is single-threaded.
- `libvmem.cuh` calls `cuCtxSynchronize` (on the current context) before every backend `move` and `destroy`, so memory never moves or disappears under a running kernel.
- When the app calls `evict(p)` or `prefetch(p)`, it promises that `p` is idle.
