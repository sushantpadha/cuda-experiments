# libvmem tests

## Where the tests are

| Path | What it holds |
|---|---|
| `tests/test1*.cu`, `test1.cuh` | test 1, the basic checks (the `.cuh` holds the checks, each `.cu` runs them on one backend) |
| `tests/test2*.cu`, `test2.cuh` | test 2, bookkeeping and error handling (same layout) |
| `tests/stress.cu`, `workload.cuh` | the stress test on one process; `workload.cuh` holds the work itself (matrix multiply, image filter) |
| `tests/daemon/` | the script-driven `vmemd` tests. Each has a C++ client and a Python checker: `state_client.cu` + `state_check.py`, `pair_client.cu` + `pair_check.py`, `worker.cu` + `workers_check.py`. `harness.py` holds what the three checkers share |
| `tests/daemon.cuh`, `utils.cuh` | C++ helpers: starting a daemon for test1d and test2d, filling and checking memory, printing results |
| `tests/data/` | five small base images (PGM) and the script that made them |

The letter at the end of a test 1 or test 2 name says which backend it uses: `c` is `cuda`, `u` is `uvm`, `v` is `vmem` without a daemon, and `d` is `vmem` with a `vmemd` daemon that the test starts itself.

## How to run them

```
make test N=1c          # also 1u 1v 1d 2c 2u 2v 2d
make stress ARGS=raw    # stress test; modes raw, cuda, uvm, vmemd; add --resident, --images N, --n MATRIX_N
make test-state         # one client against vmemd, scenario by scenario
make test-pair          # two clients against one vmemd at the same time
make test-workers       # four worker processes at once, one vmemd; ARGS="--no-resident --images 4 ..." changes it
make test-daemon        # test-state, test-pair and test-workers in that order; stops at the first failure
make daemon             # only build bin/vmemd
make clean
```

Run the stress tests with nothing else on the GPU. The `vmem` backend puts overflow into pinned host memory, which the kernel cannot swap out, so check `free -m` before a run that fills memory. With `--resident`, the stress test and `test-workers` push evicted images into pinned host memory, up to about 10 GB for the stress test. Test 2d and 2v spill up to 3 GB and 6 GB, and `test-state` and `test-pair` fill the whole VRAM once and then put 256 to 512 MiB on the host.

The Python checkers start `bin/vmemd --debug` in their own directory under `/tmp` (`vmemd-state-*`, `vmemd-pair-*`, `vmemd-workers-*`). They run every process in its own process group and kill all of them on an error, a timeout or Ctrl-C. They delete the directory when every check passes and keep it, with the broker log, when one fails. Each checker ends with a summary that lists every check as PASS or FAIL, grouped by scenario or phase.

## Tests for every backend

Tests 1 and 2 run the same code on four set-ups: `cuda`, `uvm`, `vmem` without a daemon (`vmem local`) and `vmem` with a daemon (`vmem + vmemd`). Test 1d starts `vmemd` with 5000 MiB of device and 4000 MiB of host, test 2d with 5000 MiB and 3000 MiB. Both start it with `--debug` and leave its log in `/tmp/vmemd-test.log`.

**Test 1** (`make test N=1c`, also `1u`, `1v`, `1d`) allocates 32 small objects from 512 B to 1 MiB, fills each with its own pattern and checks that no object overwrote another. It frees every other object, allocates the same sizes again and checks that the new objects reuse the freed slots without a new chunk and without damaging the live objects. Then it allocates 5 MiB and 40 MiB, fills both and checks both. A failure means the allocator hands out overlapping memory or loses track of free slots.

**Test 2** (`make test N=2c`, also `2u`, `2v`, `2d`) checks bookkeeping and error handling. It allocates 5 MiB and 100 B and checks that `used()` reports one 6 MiB chunk plus one 2 MiB chunk, and 0 after both are freed. It makes bad calls (free of a pointer inside an object, double free, free of a stack address) and checks that each throws `Unsupported` without crashing. It fills a 5 MiB allocation, evicts it to host, prefetches it back and checks every word. It allocates 256 MiB chunks until it gets `nullptr` or reaches twice the VRAM, frees them and checks that the next allocation works. Last, it launches a kernel that writes into freed memory and checks that the next `free` and every call after it throw `Fatal`. A failure means the allocator miscounts memory, accepts a bad pointer, loses data in a move, or keeps going after the GPU context died.

| What is checked | Test | cuda | uvm | vmem local | vmem + vmemd |
|---|---|---|---|---|---|
| Small objects keep their own data and do not overlap | 1 | same | same | same | same |
| Freed small slots are reused and do not damage live objects | 1 | same | same | same | same |
| Large allocations are separate and keep their data | 1 | same | same | same | same |
| The link to the daemon at the start | 1, 2 | local | local | local | connected |
| The link to the daemon at the end (a broker lost mid-test would make every other check pass without it) | 1, 2 | not checked | not checked | not checked | still connected |
| `used()` counts live chunks rounded to 2 MiB and returns to 0 after everything is freed | 2 | same | same | same | same |
| `used(Loc)` adds up to `used()` | 2 | yes | throws `Unsupported` (it cannot tell where pages are) | yes | yes |
| `alloc(0)` gives `nullptr`, `free(nullptr)` does nothing | 2 | same | same | same | same |
| Freeing the inside of an object, twice, or a foreign pointer throws `Unsupported` | 2 | same | same | same | same |
| `evict` and `prefetch` on a foreign pointer or a small object return `Unsupported` | 2 | same | same | same | same |
| `evict` and `prefetch` on a 5 MiB allocation | 2 | `Unsupported`, stays on device | `Ok`, data kept | `Ok`, data kept | `Ok`, data kept |
| A 1 PiB request gives `nullptr` | 2 | same | same | same | same |
| Allocating 256 MiB chunks until failure | 2 | `nullptr` after 22 | no failure up to twice the VRAM (45) | no failure up to twice the VRAM; the rest goes to host | `nullptr` after 30, the daemon's refusal: 4864 MiB device, 2816 MiB host |
| The allocator works again after running out | 2 | same | same | same | same |
| A kernel that writes to freed memory makes the next `free` throw `Fatal`, and every call after it | 2 | same | same | same | same |

## Tests only for vmem with a daemon

These tests compare what a client sees with what the broker counted. The broker, started with `--debug`, writes a line after every message it handles:

```
[DBG] state: pid 1234 device 12 MiB host 4 MiB, all tenants device 20 MiB host 4 MiB
[DBG] totals: all tenants device 0 MiB host 0 MiB          (after a disconnect)
```

Each client prints `view D H` after each step, with its own device and host MiB as `used(Loc)` reports them, and `WAIT x` when it needs the checker to do something. While the client has no broker it prints `local D H`, which the checkers skip. The checkers run these checks on every log:

- every view is a state the broker reached for that pid, in the same order, with the context overhead added to the device count, and the broker's last state for the pid equals the client's last view;
- after every state and totals line, the totals equal the sum over the tenants still connected;
- every request the broker refused (a log line with `refused`) left that tenant's counts unchanged;
- the `hello: pid N is "name"` line carries the name the client passed to `vmem::init`;
- after the last client exits, the last `totals:` line is 0 MiB device and 0 MiB host.

### `make test-state`

`state_check.py` runs `bin/state_client` once per scenario, each against a fresh broker. Unless a row says otherwise, the broker allows 16 MiB of device and 8 MiB of host with no context overhead, and every chunk is 4 MiB.

| Scenario | What it does | What a failure means |
|---|---|---|
| `simple` | Overhead 4 MiB. Allocates two chunks and frees them, then two 100 B objects, which share one 2 MiB chunk, and frees them. Checks that the first state is exactly the 4 MiB overhead. | The broker's counts drift from the client's after a single `Alloc` or `Free`, small objects send one `Alloc` each, or the overhead is counted twice. |
| `multi` | Fills the device (4 chunks), then the host (2 chunks). A seventh chunk gets `nullptr`. Evicts a chunk to the full host (refused), frees a host chunk, evicts again (granted), fills the freed device room, prefetches (refused), frees, prefetches (granted). After every refused move it checks `where()` and every word of the chunk. Requires `alloc of 4 MiB refused`, `move of 4 MiB to host refused` and `move of 4 MiB to device refused` in the log. | Placement is not device first and then host, a limit is not met exactly, freed room is not reused, or a refused move changed the chunk or the counts. |
| `resync` | Device 20 MiB, overhead 4 MiB. Holds two chunks; the checker kills and restarts the broker. Without a broker the client allocates a third chunk, allocates and frees a fourth, and evicts the first to host. It reconnects, then prefetches the first chunk and allocates two more, which only fit if the broker counted exactly the chunks alive. Requires exactly 3 `resync: pid N` lines and 2 `hello` lines for the pid. | The link does not turn `Broken`, the client cannot work alone, or `reconnect()` reports a freed chunk, misses a chunk made while broken, or reports the wrong location. |
| `allocfail` | No device limit, host 2048 MiB. Allocates 256 MiB chunks until one lands on host because `cuMemCreate` on the device failed. Writes and checks that chunk. Requires `alloc_fail: pid` in the log. | The fallback to host after a failed device create is broken, or the broker keeps counting that chunk on the device. |
| `tenants` | Client `hold` keeps 8 MiB. Client `fill` then gets 2 chunks on the device, 2 on the host, and `nullptr`, and checks the broker's totals in its views (12, 16 MiB device and so on). Requires `alloc of 4 MiB refused`. | The broker does not count one tenant's memory against another's limits. |
| `hang` | Holds one chunk; the checker stops the broker with `SIGSTOP`. The next allocation must return in 1 to 5 s with a chunk made locally and the link `Broken`. The checker resumes the broker and waits for it to drop the old connection; the client reconnects, and the broker counts both chunks. | A hung broker blocks the app, or the timed-out connection leaves wrong counts behind. |
| `nobroker` | Runs the client twice, with no broker: once on a socket path that does not exist, once on a stale socket file. `init` must succeed in local mode, allocation and `evict` must work, and `reconnect()` must return `Disconnected`, leave the link `Broken` and change nothing. | An app fails to start when no daemon runs. |
| `raw` | Python sends hand-made messages over the socket (overhead 4 MiB): two `Hello`s, a second `Alloc` for the same pointer, `Free`/`AllocFail`/`MoveDone`/`Move` for an unknown pointer, a second `Move` while one is in flight, `MoveDone` at the old location (failed move), `Free` during a move, `MoveDone` without a grant, `Resync` of a known pointer and past the host limit, `AllocFail` past the host limit and twice, and a 10-byte message. After each message it checks the broker's counts and requires the matching log line (`already holds; refused`, `is still moving; refused`, `failed, stays on device`, `over limit on host`, `-byte message`, ...). | One of the edge cases in `daemon/README.md` is handled differently from what it says. The real client cannot trigger most of these. |

### `make test-pair`

`pair_check.py` runs two `bin/pair_client` processes against one broker at the same time and releases them step by step. The first config allows 24 MiB of device and 12 MiB of host with 4 MiB of overhead per tenant; chunks are 4 MiB. Every phase prints the broker's per-tenant and total counts and the number of refusals, AllocFails, Resyncs and disconnects in the summary.

| Phase | What it does | What a failure means |
|---|---|---|
| 1 connect | Clients `a` and `b` connect. The broker must count 4 MiB for each and 8 MiB in total, and log the names `pair-a` and `pair-b`. | The overhead is missing or counted twice. |
| 2 share | `a` allocates 3 chunks (20 MiB in total). `b` gets one chunk on the device and the next three on the host, then `nullptr`, and its `prefetch` is refused. `a` then gets `nullptr` and its `evict` is refused; both check the chunk did not move and its data is intact. Each client must have exactly 2 refusals in the log. | A tenant does not see the other's memory, or a refusal changed something. |
| 3 reuse | `a` frees a chunk. The checker waits until the broker logged `free: pid a`, then `b` allocates and gets the freed room on the device: 24 MiB device and 12 MiB host in total, both limits met exactly. The wait is needed because `Free` has no reply: without it, `b`'s `Alloc` could reach the broker first on its own socket (see "Messages from different tenants" in `daemon/README.md`). | Freed room is not given to another tenant. |
| 4 crash | The checker kills `b` with `SIGKILL` while it holds 8 MiB device and 12 MiB host. The broker must drop it, leaving 12 MiB, `a`'s share. `a` then allocates 3 chunks on the device and one on the host, room that only exists because `b` is gone. | A crashed tenant's memory stays counted forever. |
| 5 restart | Client `c` joins and allocates two chunks. The checker kills the broker; `a` and `c` each allocate a chunk alone and `c` evicts one locally. A new broker starts with no device limit and 1024 MiB of host. Both reconnect; the broker must count `a` at 16/0 MiB, `c` at 12/4 MiB and 28/4 MiB in total, with exactly 3 `resync:` lines and 2 `hello` lines per client. | Resync after a restart loses chunks, or two tenants resyncing at once corrupt the totals. |
| 6 allocfail | `a` allocates 256 MiB chunks until one lands on host, then `c` allocates 256 MiB, which must also land on host. Requires `alloc_fail: pid` for both, and `c` counted at 12/260 MiB. | The `AllocFail` path is wrong when another tenant holds memory. |
| 7 end | Both free everything and exit. Runs the common checks over the whole log, including `b`'s views from before it was killed, and requires the broker to still run. | Something above left the broker's counts out of step. |

### `make test-workers`

`workers_check.py` starts one `vmemd` (5400 MiB device, 3000 MiB host, 100 MiB overhead per worker) and four `bin/worker` processes at the same time (no MPS). Each worker has its own matrix size (2048, 3072, 4096, 5120) and six images of 128, 192, 256 or 320 MiB, and does the work of the stress test, keeping all its images and moving them with `evict` and `prefetch`. The script first runs each worker alone to get a reference checksum, then all four together. It checks that every worker exits with 0 and no wrong spot checks, that each checksum equals the one from the run alone, that the daemon still runs and saw all four names, and the common checks on the log. It prints a table of times and the most memory the daemon counted at once. A failure means that sharing the GPU through the daemon changed a result or broke its accounting.

## Stress test results

All runs are on 2026-10-20 on the RTX 4050 Laptop (5770 MiB VRAM, 15 GB RAM), each run once.

### One process (`make stress`)

Four images of 2404 MiB each (about 5/12 of the VRAM) and 8192 × 8192 matrices. Every run gave the checksum `12b62e4bb28`. The `vmem` rows run without a daemon unless marked.

| Mode | Total | Matmul | Sharpen | Peak VRAM* | Peak DRAM* |
|---|---|---|---|---|---|
| raw | 6.99 s | 1.38 s | 5.61 s | 4916 MiB | 146 MiB |
| cuda | 2.31 s | 1.28 s | 1.04 s | 4814 MiB | 38 MiB |
| uvm | 6.15 s | 1.37 s | 4.79 s | 4914 MiB | 51 MiB |
| vmem | 2.16 s | 1.24 s | 0.92 s | 4814 MiB | 10 MiB |
| raw `--resident` | 41.21 s | 1.39 s | 39.82 s | 5666 MiB | 5710 MiB |
| uvm `--resident` | 14.18 s | 1.40 s | 12.78 s | 5666 MiB | 9320 MiB |
| vmem `--resident` | 9.60 s | 1.24 s | 8.36 s | 4830 MiB | 7264 MiB |
| vmem `--resident`, with `vmemd` | 11.41 s | 1.23 s | 10.18 s | 4828 MiB | 7015 MiB |
| cuda `--resident` | out of memory at image 2 | | | | |

\* VRAM is the use of the whole GPU and DRAM is the use of the whole machine, both minus their values at the start. Other programs shift these numbers, and a 10 ms sampler can miss short peaks.

### Four processes at once (`make test-workers`)

Each worker has six images (128, 192, 256 and 320 MiB, one size per worker). The daemon allowed 5400 MiB on the device and 3000 MiB in pinned host memory, plus 100 MiB per worker for the CUDA context. All four checksums matched the runs alone.

| Run | All four together | Most the daemon counted at once | Hints refused |
|---|---|---|---|
| one image at a time | 1.5 s | 1532 MiB device, 0 MiB host | 0 |
| all images kept, with `evict`/`prefetch` | 2.4 to 3.6 s (four runs) | 3858 to 4560 MiB device, 2944 MiB host | 10 to 13 requests refused by the daemon |

### What the results say

- **Device memory from the VMM calls is as fast as `cudaMalloc`.** Without `--resident` one image fits in the VRAM, so nothing is short of memory. `cuda` and `vmem` run the sharpen step in 1.04 s and 0.92 s. `raw` and `uvm` need 5.61 s and 4.79 s, about five times longer, because managed memory pays a page fault the first time each page is touched.
- **When the data does not fit, `cudaMalloc` fails and managed memory is slow.** With `--resident` all four images need more than the 5770 MiB of VRAM. `cuda` runs out of memory at image 2. `raw` finishes but takes 41.2 s, because the driver faults pages back and forth.
- **Moving whole images on request is the fastest way to get through.** With `--resident`, `vmem` takes 9.6 s and `uvm` with the same hints takes 14.2 s. Each `vmem` move is one copy and causes no page faults.
- **The price is host memory.** Every evicted image sits in pinned host memory until it is fetched back. `vmem` used less than `uvm`: 7264 MiB against 9320 MiB of DRAM, and 4830 MiB against 5666 MiB of VRAM. Its pinned memory does not show up in the process size, which stayed at 99 MiB.
- **Asking the daemon costs a little.** With `vmemd` (limits 5000 MiB device and 8000 MiB host, 92 MiB per client for the CUDA context) `--resident` took 11.4 s against 9.6 s without it, with the same checksum. The daemon refused 2 of the 12 hints, both evictions to a full host budget, so those two images stayed on the device. One run does not tell how much of the 1.8 s comes from the refused hints and how much from the extra messages. Without `--resident`, `vmem` with `vmemd` took 2.16 s, the same as without the daemon.
- **Four processes can share the GPU through one daemon.** Each worker's checksum matched its run alone (`1800db22fe`, `23f72bd936`, `2fd3297feb`, `3bcbd96840` for workers 0 to 3). With all images kept, the daemon counted up to 4560 MiB on the device and 2944 MiB on the host, close to its 3000 MiB host budget, and refused 10 to 13 requests. A refused hint leaves the image where it is and the worker goes on. When the workers exit, the daemon's counts return to zero.
