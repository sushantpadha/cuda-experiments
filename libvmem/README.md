# libvmem

**Status: work in progress.** Tracker 1B. A per-process allocator that apps include (`src/libvmem.cuh`) and call directly. Memory comes from one backend, chosen at `init`: `vmem` (VMM, later through the `vmemd` daemon), `uvm` or `cuda`. Only `cuda` is built so far. Next: hint-driven moves (tracker 1Biv); the hint calls exist but only mark chunks.

## Layout

```
src/libvmem.cuh     the allocator: pools, synchronization, public API (header only)
src/backend.h       backend interface and shared types (Loc, Status, Link, Chunk)
src/backends/       one header per backend: cuda.h
tests/              test programs: testN.cu
bin/                build output (not tracked)
```

## Build and run

```
make test N=1           # build and run tests/test1.cu
```

| Test | Checks |
|---|---|
| `test1.cu` | cuda backend: small objects reuse chunks and keep their data, double free caught, large allocations get one chunk each, frees release chunks, out of memory returns `nullptr` |

## API

```
vmem::init({.backend = vmem::Kind::Cuda});   // once per process; VMEM_BACKEND=vmem|uvm|cuda overrides
float *p = vmem::alloc<float>(n);            // or vmem::alloc(bytes); nullptr when out of memory
vmem::free(p);                               // block and data are dead
vmem::free(p, vmem::Reuse::Later);           // hint: I will want a block like this again
vmem::evict(p);                              // hint: p is idle, I want its data later
vmem::prefetch(p);                           // hint: I want p soon
vmem::get().info();                          // backend, daemon link, usage, chunk counts
vmem::shutdown();                            // also runs at exit
```

## Hints

Hints only mark a chunk; `libvmem.cuh` decides if and when to act, and only large allocations take part.

| Call | Meaning | State | What libvmem may do |
|---|---|---|---|
| `evict(p)` | idle, data wanted later | Cold | under VRAM pressure, move to host (same address) |
| `free(p, Reuse::Later)` | data dead, block wanted later | Cached | keep the chunk for the next fitting `alloc`; under pressure, destroy it or park it on host |
| `prefetch(p)` | wanted soon | Hot | if on host and VRAM has room, move back |

Pressure = a device `create` returns out of memory (later also a `vmemd` request for VRAM back). Moves go through backend `move`, always after `cuCtxSynchronize`. The `cuda` backend cannot move, so there hints do nothing. Stubs for all of this are in `libvmem.cuh` (`reclaim`, the `free` reuse path, `prefetch`).

Errors: `alloc` returns `nullptr` when the backend is out of memory. Misuse (a foreign pointer, a double free) throws `vmem::Error`. A fatal CUDA error (dead context) also throws, and every later call throws too.

## Pools

| | SmallPool | LargePool |
|---|---|---|
| Takes | at most 1 MiB | more than 1 MiB |
| Chunks | 2 MiB, shared | one per allocation, rounded up to 2 MiB |
| Inside a chunk | power-of-two objects of one size, at least 512 B | the whole allocation |
| On free | slot reused; empty chunk released | chunk released, or kept with `Reuse::Later` |
| Moves | never | by libvmem under pressure, guided by hints |

The thresholds follow PyTorch's caching allocator. Unlike PyTorch, the large pool caches a freed chunk only when told to (`Reuse::Later`).

## Backends

A backend sees only chunks: `create` (any multiple of its granularity; the backend picks device or host), `destroy`, `move` (same address; how libvmem carries out hints), plus `open`, `close`, `link`, `reconnect`, `usage`. Backends never synchronize.

| Backend | Memory | Moves |
|---|---|---|
| `cuda` | `cudaMalloc`, device only | no |

## Rules

- Single-threaded.
- `libvmem.cuh` calls `cuCtxSynchronize` (current context) before every backend `move` and `destroy`, so memory never moves or disappears under a running kernel.
- `evict(p)` means the app promises `p` is idle.
