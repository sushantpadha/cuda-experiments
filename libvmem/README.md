# libvmem

**Status: work in progress.** Tracker 1B. A per-process allocator that apps include (`src/libvmem.cuh`) and call directly. Memory comes from one backend, chosen at `init`: `vmem` (VMM, later through the `vmemd` daemon), `uvm` or `cuda`. `cuda` and `uvm` are built; `vmem` is not. Next: the `vmem` backend, so `evict` and `prefetch` can move memory with the address kept (tracker 1Biv).

## Layout

```
src/libvmem.cuh     the allocator: pools, synchronization, public API (header only)
src/backend.h       backend interface and shared types (Loc, Status, Link, Chunk)
src/backends/       one header per backend: cuda.h, uvm.h
tests/              testNb.cu mains (N = test, b = backend: c cuda, u uvm, v vmem); testN.cuh bodies; common.cuh helpers
bin/                build output (not tracked)
```

## Build and run

```
make test N=1c         # build and run tests/test1c.cu; also 1u, 2c, 2u
```

Each test has one main per backend (`c` cuda, `u` uvm, `v` vmem) that runs the same body. `v` fails at `init` until the vmem backend exists.

## Tests

Add a row when a test starts checking a new behaviour. `c`/`u` = what the cuda/uvm backend is expected to do where it differs.

| Behaviour | Test |
|---|---|
| Small objects keep their own data, none overlaps another | 1 |
| Freed small slots are reused: no new chunk, live objects not clobbered | 1 |
| Large allocations are separate and keep their data | 1 |
| `used()` counts live chunks rounded to 2 MiB (large 5 MiB = 6 MiB; small object = one 2 MiB chunk) | 2 |
| `used()` returns to 0 after everything is freed | 2 |
| `used(Loc)`: device + host = total (c); throws `Unsupported` (u) | 2 |
| `info()` chunk counts per pool | 2 |
| `alloc(0)` returns `nullptr`; `free(nullptr)` does nothing | 2 |
| `free` of an interior pointer, a double free, or a foreign pointer throws `Unsupported` | 2 |
| `small_max` larger than `small_chunk` throws `Unsupported` | 2 |
| `evict`/`prefetch` on a foreign pointer or small object return `Unsupported`, no throw | 2 |
| `evict`/`prefetch` on a large allocation: `Unsupported`, stays on device (c); `Ok`, `where()` follows, data intact (u) | 2 |
| Absurd request (1 PiB) returns `nullptr` | 2 |
| Out of VRAM returns `nullptr` (c); no failure up to 2x VRAM (u) | 2 |
| Allocator works again after out of memory | 2 |
| A faulted kernel (write to freed memory) makes the next `free` throw `Fatal`, and every later call too (GPU logs an Xid 31, recoverable) | 2 |

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

`alloc` and `free` take no hints. The only additions are `evict(p)` and `prefetch(p)`, which move one large allocation now: `libvmem.cuh` calls `cuCtxSynchronize`, then backend `move` (same address, data kept). The app promises `p` is idle. Small objects share chunks and never move.

Hints block until the copy is done. Non-blocking hints are a TODO (tracker 1Biv): the copy can run on its own stream, but the swap of the old backing for the new one (`cuMemUnmap` + `cuMemMap`) is a host call that no event or stream can order, so the app would have to call `wait(p)` before touching `p` again.

A hint cannot fail the program. It returns a `Status`, and on anything but `Ok` the chunk stays where it was:

| Status | Meaning |
|---|---|
| `Ok` | the chunk is at the target (also when it already was) |
| `NoMemory` | no room at the target |
| `Unsupported` | small object, pointer not from libvmem, or the backend cannot move (`cuda`) |
| `Fatal` | the CUDA context is dead; later `alloc` and `free` throw |

Errors: `alloc` returns `nullptr` when the backend is out of memory. Misuse (a foreign pointer, a double free) throws `vmem::Error`. A fatal CUDA error (dead context) also throws, and every later call throws too.

## Pools

| | SmallPool | LargePool |
|---|---|---|
| Takes | at most 1 MiB | more than 1 MiB |
| Chunks | 2 MiB, shared | one per allocation, rounded up to 2 MiB |
| Inside a chunk | power-of-two objects of one size, at least 512 B | the whole allocation |
| On free | slot reused; empty chunk released | chunk released |
| Moves | never | `evict` and `prefetch` |

The thresholds follow PyTorch's caching allocator. Unlike PyTorch, the large pool does not cache freed chunks.

## Backends

A backend sees only chunks: `create` (any multiple of its granularity; the backend picks device or host), `destroy`, `move` (same address; on failure the chunk is unchanged), plus `open`, `close`, `link`, `reconnect`, `used`. Backends never synchronize.

| Backend | Memory | Moves |
|---|---|---|
| `cuda` | `cudaMalloc`, device only | no |
| `uvm` | `cudaMallocManaged`; `move` is a blocking `cudaMemPrefetchAsync` | yes, but `Ok` means requested, not resident |

On `uvm` the driver decides where pages live, so `where()` reports the last requested location and `used(Loc)` throws `Unsupported`.

## Rules

- Single-threaded.
- `libvmem.cuh` calls `cuCtxSynchronize` (current context) before every backend `move` and `destroy`, so memory never moves or disappears under a running kernel.
- `evict(p)` and `prefetch(p)` mean the app promises `p` is idle.
