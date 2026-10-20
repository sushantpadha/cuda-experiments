# libvmem tests

## Build and run

```
make test N=1c         # build and run tests/test1c.cu; also 1u, 1v, 2c, 2u, 2v
make test N=4          # vmemd accounting; builds and starts bin/vmemd itself
make test N=5          # the vmem backend as a vmemd client; starts bin/vmemd itself
```

The `vmem` tests connect to the broker at `/tmp/vmemd.sock` if one is running there, and run in local mode otherwise. Set `VMEM_DAEMON=PATH` to use another socket. Tests 4 and 5 start their own broker on `/tmp/vmemd-test.sock` and ignore `VMEM_DAEMON`.

Each test has one main per backend (`c` for cuda, `u` for uvm, `v` for vmem), and all of them run the same test body. Each main passes its own name (for example `test1u`) to `vmem::init` as `Config::name`, so the broker's log shows which test a tenant is.

Test 3 is a stress test with four modes. `raw` uses plain `cudaMallocManaged`, and `cuda`, `uvm` and `vmemd` go through libvmem. The `vmemd` mode runs the `vmem` backend, which talks to a running broker as described above. Test 3 does not start one.

```
make test N=3 ARGS=raw                  # tiled matrix multiply + sharpen on 4 images of 5/12 VRAM each
make test N=3 ARGS=cuda                 # same through libvmem; compare time, peak VRAM, peak DRAM, checksum
make test N=3 ARGS="uvm --resident"     # all images live at once, evict/prefetch around each use
```

The flags are `--resident`, `--images N` (default 4), `--n MATRIX_N` (default 8192), and `--data DIR`. The checksum must be the same in every mode.

The test samples peak memory every 10 ms across the whole device and the whole system, so run it with nothing else on the GPU.

With `--resident`, all images stay allocated and need more memory than the GPU has. `cuda` stops with OUT OF MEMORY, as expected. The other modes push evicted images into DRAM, up to about 10 GB on this machine, so check `free -m` first.

## Tests

When a test starts checking a new behaviour, add a row here. Where the backends differ, `(c)` and `(u)` mark what the cuda and uvm backends are expected to do.

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
| Out of VRAM returns `nullptr` (c); no failure up to 2x VRAM (u, v) | 2 |
| Allocator works again after out of memory | 2 |
| Tiled matmul and sharpen on 4 images of 5/12 VRAM: results match across modes, spot checks pass | 3 |
| A faulted kernel (write to freed memory) makes the next `free` throw `Fatal`, and every later call too (GPU logs an Xid 31, recoverable) | 2 |
| `ctx_overhead_mib` counts once per tenant on its first `Hello` and goes away with the tenant | 4 |
| `vmemd` refuses to start without a config file, or with an unknown key, a value below -1, or trailing text | 4 |
| A missing limit or -1 means unlimited | 4 |
| `Alloc` goes to the device until the VRAM limit, then to host until the DRAM limit, then gets `NoMemory`; a repeated pointer gets `Unsupported` | 4 |
| `AllocFail` moves a chunk's bytes from device to host; unknown and host chunks are ignored | 4 |
| `Free` returns the bytes at the chunk's location; unknown pointers and double frees change nothing | 4 |
| A tenant's usage goes away when its socket closes | 4 |
| `Move` reserves the target until `MoveDone`; one move per tenant at a time; a failed move or a free during a move releases the reservation | 4 |
| `Resync` is accepted over a limit and overwrites a known pointer; `MoveDone` without a grant is trusted | 4 |
| 3000 random calls from 3 tenants match a model of the broker's rules, including `NoMemory`; after freeing everything the limits are met exactly | 4 |
| No broker at `init`: link `Local`, chunks on device | 5 |
| With a broker, its limits decide placement, not free VRAM; host chunks keep their data; both limits full gives `nullptr` | 5 |
| `prefetch`/`evict` without room at the target return `NoMemory` and leave the chunk; after a free they succeed with data intact | 5 |
| Broker killed: link `Broken`, new chunks go to the device locally; `reconnect()` without a broker returns `Disconnected` | 5 |
| Broker restarted: `reconnect()` returns `Ok`, and the resynced chunks (including local ones) count against the limits | 5 |
| Broker hung (`SIGSTOP`): an `alloc` returns after the 2 s timeout, served locally, link `Broken` | 5 |
| Device `cuMemCreate` fails after the broker said device: the chunk lands on host and the broker logs `AllocFail` | 5 |

## Stress test results

We ran `test3` on 2026-10-20 on an RTX 4050 Laptop (5770 MiB VRAM) with 15 GB RAM, using 4 images of 2404 MiB each and 8192² matrices. Each mode ran once, and every run produced the checksum `12b62e4bb28`. The `vmem` rows use local mode, except the one marked "with `vmemd`".

| Mode | Total | Matmul | Sharpen | Peak VRAM* | Peak DRAM* |
|---|---|---|---|---|---|
| raw | 6.99 s | 1.38 s | 5.61 s | 4916 MiB | 146 MiB |
| cuda | 2.31 s | 1.28 s | 1.04 s | 4814 MiB | 38 MiB |
| uvm | 6.15 s | 1.37 s | 4.79 s | 4914 MiB | 51 MiB |
| vmem (local, no `vmemd`) | 2.16 s | 1.24 s | 0.92 s | 4814 MiB | 10 MiB |
| raw `--resident` | 41.21 s | 1.39 s | 39.82 s | 5666 MiB | 5710 MiB |
| uvm `--resident` | 14.18 s | 1.40 s | 12.78 s | 5666 MiB | 9320 MiB |
| vmem `--resident` (local, no `vmemd`) | 9.60 s | 1.24 s | 8.36 s | 4830 MiB | 7264 MiB |
| vmem `--resident` (with `vmemd`) | 11.41 s | 1.23 s | 10.18 s | 4828 MiB | 7015 MiB |
| cuda `--resident` | out of memory at image 2 | | | | |

\* VRAM is device-wide use and DRAM is system-wide use (from `/proc/meminfo`), both minus their values at the start. Other processes shift these numbers, and the 10 ms sampler can miss short spikes, so treat them as rough.

What the results show:

- Without `--resident`, one image fits in VRAM and nothing is oversubscribed. `cuda` and `vmem` run the sharpen phase about 5x faster than `raw` and `uvm`, because managed memory pays a page-fault cost on first touch. VMM memory on the device behaves like `cudaMalloc`.
- With `--resident`, `cudaMalloc` cannot hold all 4 images, so `cuda` runs out of memory.
- Hints help. With `--resident`, `vmem` takes 9.6 s and `uvm` takes 14.2 s, against 41.2 s for `raw`, where the driver faults pages back and forth. `vmem` moves each chunk with one copy and never faults.
- With `vmemd` (`daemon/vmemd.toml`: 5000 MiB VRAM, 8000 MiB DRAM, 92 MiB context overhead), the broker refused 2 of 12 hints, both evictions to a full DRAM budget, so those images stayed on the device. The run took 11.4 s against 9.6 s in local mode, with the same checksum. This single run does not tell how much of the gap comes from the refused hints and how much from the round trips. Without `--resident`, `vmem` with `vmemd` took 2.16 s, the same as in local mode.
- Hints cost DRAM, because every evicted image lands in host memory. `vmem` used less than `uvm`: 7264 MiB against 9320 MiB of DRAM, and 4830 MiB against 5666 MiB of VRAM. Its host memory is pinned and does not show up in the process RSS, which stayed at 99 MiB.
