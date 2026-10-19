# libvmem

**Status: work in progress.** Tracker 1B. A per-process allocator that apps include (`src/libvmem.cuh`) and call directly. Memory comes from one backend, chosen at `init`: `vmem` (VMM, later through the `vmemd` daemon), `uvm` or `cuda`. `cuda` and `uvm` are built; `vmem` is not. Next: the `vmem` backend, so `evict` and `prefetch` can move memory with the address kept (tracker 1Biv).

## Layout

```
src/libvmem.cuh     the allocator: pools, synchronization, public API (header only)
src/backend.h       backend interface and shared types (Loc, Status, Link, Chunk)
src/backends/       one header per backend: cuda.h, uvm.h
tests/              testNb.cu mains (N = test, b = backend: c cuda, u uvm, v vmem); testN.cuh bodies; common.cuh helpers
tests/data/         small base images (PGM) and make_images.py; test3 tiles them up to VRAM-sized images
bin/                build output (not tracked)
```

## Build and run

```
make test N=1c         # build and run tests/test1c.cu; also 1u, 2c, 2u
```

Stress test (test3): `raw` is plain `cudaMallocManaged` without libvmem; `cuda`, `uvm`, `vmemd` go through libvmem.

```
make test N=3 ARGS=raw                  # tiled matrix multiply + sharpen on 4 images of 5/12 VRAM each
make test N=3 ARGS=cuda                 # same through libvmem; compare time, peak VRAM, peak DRAM, checksum
make test N=3 ARGS="uvm --resident"     # all images live at once, evict/prefetch around each use
```

Flags: `--resident`, `--images N` (4), `--n MATRIX_N` (8192), `--data DIR`. Peaks are sampled every 10 ms: VRAM is device-wide used and DRAM is system-wide used, both minus the value at start, so run it with nothing else on the GPU. The checksum must match across modes. `--resident` needs more than the VRAM: `cuda` stops with OUT OF MEMORY (expected), and `uvm` and `raw` push the evicted images into DRAM (up to about 10 GB here).

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
| Tiled matmul and sharpen on 4 images of 5/12 VRAM: results match across modes, spot checks pass | 3 |
| A faulted kernel (write to freed memory) makes the next `free` throw `Fatal`, and every later call too (GPU logs an Xid 31, recoverable) | 2 |

## Stress test results

`test3` on 2026-10-20: RTX 4050 Laptop (5770 MiB VRAM), 15 GB RAM, 4 images of 2404 MiB, matrices 8192². One run each, `cuda` and `uvm` are the libvmem backends, `raw` is plain `cudaMallocManaged`. All checksums are `12b62e4bb28`.

| Mode | Total | Matmul | Sharpen | Peak VRAM* | Peak DRAM* |
|---|---|---|---|---|---|
| raw | 6.99 s | 1.38 s | 5.61 s | 4916 MiB | 146 MiB |
| cuda | 2.31 s | 1.28 s | 1.04 s | 4814 MiB | 38 MiB |
| uvm | 6.15 s | 1.37 s | 4.79 s | 4914 MiB | 51 MiB |
| raw `--resident` | 41.21 s | 1.39 s | 39.82 s | 5666 MiB | 5710 MiB |
| uvm `--resident` | 14.18 s | 1.40 s | 12.78 s | 5666 MiB | 9320 MiB |
| cuda `--resident` | out of memory at image 2 | | | | |

\* Peaks are sampled every 10 ms. VRAM is device-wide used and DRAM is system-wide used (`/proc/meminfo`), both minus the value at start. Any other process using the GPU or allocating memory during the run shifts them, and the sampler can miss a spike shorter than 10 ms. Treat them as rough.

Notes:

- Without `--resident` one image (input and output) fits in VRAM, so nothing oversubscribes. `cuda` is about 3x faster than `raw` and `uvm` on the sharpen phase because managed memory pays page-fault cost on first touch.
- `cuda --resident` stops with out of memory: `cudaMalloc` cannot hold 4 images.
- Hints help: with `--resident`, `uvm` with `evict` and `prefetch` takes 14 s against 41 s for `raw`, where the driver faults pages back and forth.
- Hints cost DRAM: every evicted image lands in host memory, so `uvm --resident` reached 9.7 GB RSS on a machine with about 6.5 GB free (other processes held the rest). Check `free -m` first.
- `vmem` backend (`vmemd`) is not in the table: it is not built yet.

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
