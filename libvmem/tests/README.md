# libvmem tests

This directory holds the tests for the allocator, its three backends and the `vmemd` broker. The measured results are in [RESULTS.md](RESULTS.md).

## What each test is for

| Test | Question it answers | Processes | Daemon |
|---|---|---|---|
| test 1 | Does the allocator hand out separate memory that keeps its data, and reuse freed slots? | 1 | only in `1d` |
| test 2 | Does the allocator count memory right, refuse bad calls, move data intact, and stop after the GPU context dies? | 1 | only in `2d` |
| stress | How fast is each backend when the data fits in VRAM and when it does not, and do they all compute the same result? | 1 | only in mode `vmemd` |
| `test-state` | Do the broker's counts match what one client sees, through limits, refusals, crashes, restarts and bad input? | 1 (2 in one scenario) | yes |
| `test-pair` | Does each of two clients see the other's memory in the limits, and do the counts stay right when one crashes or the broker restarts? | 2 | yes |
| `test-workers-gemm` | Do four programs sharing the GPU through the daemon still compute the right answer when the daemon places some of their memory on the host? | 4 | yes, or none for the baselines |
| `test-workers-image` | Do they still compute the right answer when they keep evicting and prefetching, and the daemon grants or refuses each move? | 4 | yes, or none for the baselines |

The tests fall into three groups. Tests 1 and 2 check the allocator. They run the same code on every backend, so the table under them shows where the backends differ. The stress test measures speed. It fails only on a crash, a wrong spot check or a checksum that differs between modes; you read its timings yourself. The daemon tests check the broker by comparing its log with what the clients saw. The `state` and `pair` tests use 4 MiB chunks and limits of a few MiB, so we know every count before the run. The two `workers` tests use real kernels at real sizes, so they check the same rules under load.

## Where the tests are

| Path | What it holds |
|---|---|
| `tests/test1*.cu`, `test1.cuh` | test 1; the `.cuh` holds the checks and each `.cu` runs them on one backend |
| `tests/test2*.cu`, `test2.cuh` | test 2, laid out the same way |
| `tests/stress.cu`, `workload.cuh` | the single-process stress test; `workload.cuh` holds the work itself (matrix multiply, image filter) |
| `tests/daemon/` | the daemon tests: `state_client.cu` + `state_check.py`, `pair_client.cu` + `pair_check.py`, and `worker.cu` + `workers_check.py`, which run as both the gemm and the image test; `harness.py` holds the code the checkers share |
| `tests/daemon.cuh`, `utils.cuh` | C++ helpers that start a daemon for test1d and test2d, fill and check memory, and print results |
| `tests/data/` | five small base images (PGM) and the script that made them |

The letter at the end of a test 1 or test 2 name says which backend it uses: `c` is `cuda`, `u` is `uvm`, `v` is `vmem` without a daemon, and `d` is `vmem` with a `vmemd` daemon that the test starts itself.

## How to run them

```
make test N=1c          # also 1u 1v 1d 2c 2u 2v 2d
make stress ARGS=raw    # modes raw, cuda, uvm, vmemd; add --resident, --images N, --n MATRIX_N
make test-state         # one client against vmemd, scenario by scenario
make test-pair          # two clients against one vmemd at the same time
make test-workers-gemm  # four processes, one matrix multiply each; ARGS="--backend uvm ..." changes it
make test-workers-image # four processes, sharpen on images; ARGS="--no-resident --images 4 ..."
make test-daemon        # test-state, test-pair, test-workers-gemm, test-workers-image; stops at the first failure
make trace-gemm         # test-workers-gemm under scripts/memtrace.py; BACKEND=vmem|vmem-local|cuda|uvm, TRACE_ARGS="--interval 0.05"
make trace-image        # the same for test-workers-image
make trace-all          # both workloads on all four backends, then plot-traces
make plot-traces        # analysis/gemm.png and analysis/image.png from the CSVs (scripts/memplot.py)
make daemon             # only build bin/vmemd
make clean
```

Run the stress tests with nothing else on the GPU. The `vmem` backend puts overflow into pinned host memory, which the kernel cannot swap out, so check `free -m` before a run that fills memory. With `--resident`, the stress test pushes up to about 10 GB of evicted images into pinned host memory, and `test-workers-image` up to 3 GB. Tests 2d and 2v spill up to 3 GB and 6 GB. `test-state` and `test-pair` fill the whole VRAM once and then put 256 to 512 MiB on the host.

The Python checkers start `bin/vmemd --debug` in their own directory under `/tmp` (`vmemd-state-*`, `vmemd-pair-*`, `vmemd-gemm-*`, `vmemd-image-*`). They run every process in its own process group and kill all of them on an error, a timeout or Ctrl-C. When every check passes they delete the directory; when one fails they keep it, with the broker log, so you can read what happened. Each checker ends with a summary that lists every check as PASS or FAIL, grouped by scenario or phase.

## Tests 1 and 2: every backend

Tests 1 and 2 run the same code on four set-ups: `cuda`, `uvm`, `vmem` without a daemon (`vmem local`) and `vmem` with a daemon (`vmem + vmemd`). Test 1d starts `vmemd` with 5000 MiB of device and 4000 MiB of host, and test 2d with 5000 MiB and 3000 MiB. Both start it with `--debug` and leave its log in `/tmp/vmemd-test.log`.

**Test 1** (`make test N=1c`, also `1u`, `1v`, `1d`) checks that the allocator hands out memory that does not overlap. It allocates 32 small objects from 512 B to 1 MiB, fills each with its own pattern and checks that no object overwrote another. It then frees every other object, allocates the same sizes again, and checks that the new objects reuse the freed slots without a new chunk and without damaging the live objects. Last, it allocates 5 MiB and 40 MiB, fills both and checks both. A failure means the allocator hands out overlapping memory or loses track of its free slots.

**Test 2** (`make test N=2c`, also `2u`, `2v`, `2d`) checks bookkeeping and error handling. It allocates 5 MiB and 100 B and checks that `used()` reports one 6 MiB chunk plus one 2 MiB chunk, then 0 once both are freed. It makes bad calls (a free of a pointer inside an object, a double free, a free of a stack address) and checks that each one throws `Unsupported` without crashing. It fills a 5 MiB allocation, evicts it to host, prefetches it back and checks every word. It allocates 256 MiB chunks until it gets `nullptr` or reaches twice the VRAM, frees them, and checks that the next allocation works. Last, it launches a kernel that writes into freed memory and checks that the next `free`, and every call after it, throws `Fatal`. A failure means the allocator miscounts memory, accepts a bad pointer, loses data in a move, or carries on after the GPU context died.

This table shows what each backend does for every check. "Same" means all four behave alike.

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
| Allocating 256 MiB chunks until failure | 2 | `nullptr` once the VRAM is full | no failure up to twice the VRAM | no failure up to twice the VRAM; the rest goes to host | `nullptr` once the daemon's limits are reached |
| The allocator works again after running out | 2 | same | same | same | same |
| A kernel that writes to freed memory makes the next `free` throw `Fatal`, and every call after it | 2 | same | same | same | same |

## Stress test (`make stress`)

The stress test compares the backends on one process, first when the data fits in VRAM and then when it does not. It also checks that every backend computes the same result. `stress.cu` runs two phases, and both live in `workload.cuh`.

1. **Matrix multiply.** It allocates three float matrices of 8192 × 8192 (256 MiB each; `--n` changes the size). It fills A[r][c] = (r + 2c) mod 5 and B[r][c] = (3r + c) mod 4. These are small integers, so every sum in C = A × B is an exact float. The kernel is the shared-memory tiled multiply from the CUDA C++ Programming Guide, with 32 × 32 tiles: each block loads one tile of A and one of B into shared memory, then accumulates. The phase checks 12 entries of C against a dot product computed on the host.
2. **Sharpen.** It makes four images (`--images` changes the count), each 16384 pixels wide and 5/12 of the VRAM in size (2404 MiB on this GPU), so one image fits and three do not. Each image is one of the five base pictures in `tests/data/`, tiled up to that size with a shift per tile so that no two tiles are equal. The 3 × 3 sharpen sets each output pixel to 5 times the centre minus its four neighbours, clamped to 0..255. The phase checks 64 output pixels against the same filter on the host and adds every output pixel to the checksum.

Without `--resident` the test makes and frees the images one at a time, so the work always fits. With `--resident` all images live at once. The test evicts each one to host right after making it, prefetches it before its sharpen, and evicts it again afterwards. That needs more than the VRAM, so this mode shows how each backend copes with too little memory.

The mode picks where the memory comes from. `raw` uses plain `cudaMallocManaged`, with no libvmem and no hints. `cuda`, `uvm` and `vmemd` use libvmem on that backend; `vmemd` is the `vmem` backend, and it uses a daemon on `/tmp/vmemd.sock` if one is running. The test prints the time of each phase, the peak VRAM and DRAM use of the whole machine (sampled every 10 ms), the peak size of its own process, the hints taken and refused, and the checksum. The checksum must be the same in every mode. A different checksum means a backend lost or damaged data.

## Tests for the daemon

These tests compare what a client sees with what the broker counted. With `--debug`, the broker writes a line after every message it handles:

```
[DBG] state: pid 1234 device 12 MiB host 4 MiB, all tenants device 20 MiB host 4 MiB
[DBG] totals: all tenants device 0 MiB host 0 MiB          (after a disconnect)
```

After each step a client prints `view D H`, its own device and host MiB as `used(Loc)` reports them. It prints `WAIT x` when it needs the checker to do something, such as restart the broker. While a client has no broker it prints `local D H`, and the checkers skip those lines. The checkers run these checks on every log:

- every view is a state the broker reached for that pid, in the same order, with the context overhead added to the device count, and the broker's last state for the pid equals the client's last view;
- after every state and totals line, the totals equal the sum over the tenants still connected;
- every request the broker refused (a log line containing `refused`) left that tenant's counts unchanged;
- the `hello: pid N is "name"` line carries the name the client passed to `vmem::init`;
- after the last client exits, the last `totals:` line reads 0 MiB device and 0 MiB host.

### `make test-state`

This test checks each rule of the broker with one client at a time, at sizes where we know every count in advance. Each scenario targets one rule from `daemon/README.md`, and the last column of the table says which rule a failure breaks.

`state_check.py` runs `bin/state_client` once per scenario, each time against a fresh broker. Unless a row says otherwise, the broker allows 16 MiB of device and 8 MiB of host with no context overhead, and every chunk is 4 MiB.

| Scenario | What it does | What a failure means |
|---|---|---|
| `simple` | Overhead 4 MiB. Allocates two chunks and frees them, then two 100 B objects, which share one 2 MiB chunk, and frees them. Checks that the first state is exactly the 4 MiB overhead. | The broker's counts drift from the client's after a single `Alloc` or `Free`, small objects send one `Alloc` each, or the overhead is counted twice. |
| `multi` | Fills the device (4 chunks), then the host (2 chunks). A seventh chunk gets `nullptr`. Evicts a chunk to the full host (refused), frees a host chunk, evicts again (granted), fills the freed device room, prefetches (refused), frees, prefetches (granted). After every refused move it checks `where()` and every word of the chunk. Requires `alloc of 4 MiB refused`, `move of 4 MiB to host refused` and `move of 4 MiB to device refused` in the log. | Placement is not device first and then host, a limit is not met exactly, freed room is not reused, or a refused move changed the chunk or the counts. |
| `resync` | Device 20 MiB, overhead 4 MiB. Holds two chunks; the checker kills and restarts the broker. Without a broker the client allocates a third chunk, allocates and frees a fourth, and evicts the first to host. It reconnects, then prefetches the first chunk and allocates two more, which only fit if the broker counted exactly the chunks alive. Requires exactly 3 `resync: pid N` lines and 2 `hello` lines for the pid. | The link does not turn `Broken`, the client cannot work alone, or `reconnect()` reports a freed chunk, misses a chunk made while broken, or reports the wrong location. |
| `autoreconnect` | Like `resync`, but the client sets `auto_reconnect` and never calls `reconnect()`. It holds two chunks; the checker restarts the broker. The next chunk is made locally and the link turns `Broken`. The chunk after that must reconnect by itself, so the new broker counts four chunks (16 MiB) and sends the fifth to host. Requires 3 `resync: pid N` lines. | `auto_reconnect` does not retry, or the retry loses a chunk or counts one twice. |
| `allocfail` | No device limit, host 2048 MiB. Allocates 256 MiB chunks until one lands on host because `cuMemCreate` on the device failed. Writes and checks that chunk. Requires `alloc_fail: pid` in the log. | The fallback to host after a failed device create is broken, or the broker keeps counting that chunk on the device. |
| `tenants` | Client `hold` keeps 8 MiB. Client `fill` then gets 2 chunks on the device, 2 on the host, and `nullptr`, and checks the broker's totals in its views (12, 16 MiB device and so on). Requires `alloc of 4 MiB refused`. | The broker does not count one tenant's memory against another's limits. |
| `hang` | Holds one chunk; the checker stops the broker with `SIGSTOP`. The next allocation must return in 1 to 5 s with a chunk made locally and the link `Broken`. The checker resumes the broker and waits for it to drop the old connection; the client reconnects, and the broker counts both chunks. | A hung broker blocks the app, or the timed-out connection leaves wrong counts behind. |
| `nobroker` | Runs the client twice with no broker: once on a socket path that does not exist, once on a stale socket file. `init` must succeed in local mode, allocation and `evict` must work, and `reconnect()` must return `Disconnected`, leave the link `Broken` and change nothing. | An app fails to start when no daemon runs. |
| `config` | Starts `vmemd` four times with a bad config: a missing file, an unknown key (`foo_mib`), `vram_limit_mib = -2`, and `vram_limit_mib = 16 MiB`. Each must exit with an error and log the reason. | The broker starts on a config it should refuse, or fails without saying why. |
| `raw` | Python sends hand-made messages over the socket (overhead 4 MiB): two `Hello`s, a second `Alloc` for the same pointer, `Free`/`AllocFail`/`MoveDone`/`Move` for an unknown pointer, a second `Move` while one is in flight, `MoveDone` at the old location (failed move), `Free` during a move, `MoveDone` without a grant, `Resync` of a known pointer and past the host limit, `AllocFail` past the host limit and twice, and a 10-byte message. After each message it checks the broker's counts and requires the matching log line (`already holds; refused`, `is still moving; refused`, `failed, stays on device`, `over limit on host`, `-byte message`, ...). | One of the edge cases in `daemon/README.md` is handled differently from what it says. The real client cannot trigger most of these. |

### `make test-pair`

This test checks the rules that only matter with more than one tenant. One tenant's memory must count against the other's limits, freed room must pass from one to the other, a crashed tenant must be dropped, and two tenants must be able to resync at once after a broker restart.

`pair_check.py` runs two `bin/pair_client` processes against one broker at the same time and releases them step by step. The first config allows 24 MiB of device and 12 MiB of host, with 4 MiB of overhead per tenant, and every chunk is 4 MiB. The summary of each phase shows the broker's per-tenant and total counts and the number of refusals, AllocFails, Resyncs and disconnects.

| Phase | What it does | What a failure means |
|---|---|---|
| 1 connect | Clients `a` and `b` connect. The broker must count 4 MiB for each and 8 MiB in total, and log the names `pair-a` and `pair-b`. | The overhead is missing or counted twice. |
| 2 share | `a` allocates 3 chunks (20 MiB in total). `b` gets one chunk on the device and the next three on the host, then `nullptr`, and its `prefetch` is refused. `a` then gets `nullptr` and its `evict` is refused; both check the chunk did not move and its data is intact. Each client must have exactly 2 refusals in the log. | A tenant does not see the other's memory, or a refusal changed something. |
| 3 reuse | `a` frees a chunk. The checker waits until the broker logged `free: pid a`, then `b` allocates and gets the freed room on the device: 24 MiB device and 12 MiB host in total, both limits met exactly. The wait is needed because `Free` has no reply: without it, `b`'s `Alloc` could reach the broker first on its own socket (see "Messages from different tenants" in `daemon/README.md`). | Freed room is not given to another tenant. |
| 4 crash | The checker kills `b` with `SIGKILL` while it holds 8 MiB device and 12 MiB host. The broker must drop it, leaving 12 MiB, `a`'s share. `a` then allocates 3 chunks on the device and one on the host, room that only exists because `b` is gone. | A crashed tenant's memory stays counted forever. |
| 5 restart | Client `c` joins and allocates two chunks. The checker kills the broker; `a` and `c` each allocate a chunk alone and `c` evicts one locally. A new broker starts with no device limit and 1024 MiB of host. Both reconnect; the broker must count `a` at 16/0 MiB, `c` at 12/4 MiB and 28/4 MiB in total, with exactly 3 `resync:` lines and 2 `hello` lines per client. | Resync after a restart loses chunks, or two tenants resyncing at once corrupt the totals. |
| 6 allocfail | `a` allocates 256 MiB chunks until one lands on host, then `c` allocates 256 MiB, which must also land on host. Requires `alloc_fail: pid` for both, and `c` counted at 12/260 MiB. | The `AllocFail` path is wrong when another tenant holds memory. |
| 7 end | Both free everything and exit. Runs the common checks over the whole log, including `b`'s views from before it was killed, and requires the broker to still run. | Something above left the broker's counts out of step. |

### `make test-workers-gemm` and `make test-workers-image`

Both tests run four processes on the GPU at the same time, without MPS, and each process is a tenant of one `vmemd`. They check that every program still computes the right answer while it shares the GPU through the daemon, and that the daemon's counts stay exact. We split them into two tests because they load the daemon in different ways. The GEMM test asks for a few large chunks once and holds them, so the daemon has to place them, and some go to host memory when the device is full. The image test allocates, evicts and prefetches many times, so the daemon has to grant or refuse every move.

Both workloads are the two halves of the stress test above. They live in `tests/workload.cuh`, so `stress.cu` and `tests/daemon/worker.cu` run the same code with the same kernels, inputs and spot checks. We picked them because they are simple, check their own results, and use memory in two different patterns. Nothing in them comes from an outside benchmark suite.

`workers_check.py` starts `vmemd --debug` with 5400 MiB of device, 3000 MiB of pinned host memory and 100 MiB of context overhead per tenant. It first runs each worker alone to record a reference checksum; `--no-solo` skips this pass. It then starts four `bin/worker` processes. Each one connects, says `Hello` with its name (`worker0` to `worker3`) and prints `READY`. When all four are ready, the script releases them together so that their work overlaps. It then checks that:

- every worker exits with 0 and got no spot check wrong;
- each worker's checksum equals its checksum from the run alone, so sharing the GPU did not change any result;
- `vmemd` still runs, saw all four names, kept its totals equal to the sum over its tenants after every message, left the counts unchanged on every refusal, and is back to 0 once the workers are gone.

At the end it prints each worker's time and hint counts, the wall time of all four together, and the most memory the daemon counted at once.

**`make test-workers-gemm`.** Worker *i* allocates three float matrices A, B and C of n × n, with n = 8192, 10240, 12288 and 14336 for workers 0 to 3. That is 256, 400, 576 and 784 MiB per matrix, and 6048 MiB for all twelve. It fills A and B as in the stress test, computes C = A × B, then reads 12 entries of C back (both corners and 10 spread out) and compares each with a dot product computed on the host. Its checksum is the sum of those 12 entries as read from the GPU. It makes no `evict` or `prefetch` calls. The twelve matrices do not fit in the 5400 MiB the daemon hands out on the device, so the daemon places the last chunks in host memory, and the kernels that use them read over PCIe. A failure means that a matrix the daemon placed gave a wrong product, that a tenant's result depended on its neighbours, or that the daemon miscounted large chunks. `--matrix-n` changes the sizes.

**`make test-workers-image`.** Worker *i* makes six images of 128, 192, 256 or 320 MiB, 16384 pixels wide, starting at base image *i* so that the workers use different pictures. By default (`--resident`) it first creates all six and evicts each one to host right after writing it. Then, for each image, it prefetches the image to the device, allocates an output image, runs the sharpen, checks 64 output pixels against the same filter on the host, adds the output's pixel sum to its checksum, frees the output and evicts the input again. Together the four workers want about 5.4 GB of images, so the daemon refuses some hints once its host budget is full. A refused hint leaves the image where it is, and the worker carries on. With `--no-resident` each worker makes one image at a time, frees it after use, and gives no hints. A failure means that a move the daemon granted lost or damaged data, or that the daemon miscounted moves.

**Baselines.** `--backend vmem-local`, `--backend cuda` or `--backend uvm` runs the same workers without a daemon, so you can compare them with `vmem` and `vmemd`. `vmem-local` is the `vmem` backend in local mode: it places chunks on the device until `cuMemCreate` fails and then on the host, and it grants every hint. The script skips the broker checks for these runs and keeps the exit code, spot check and checksum checks. A worker that runs out of memory exits with 3 and fails its check. That is a result of the run, not an error in the harness.

Every worker prints `hint ok` or `hint refused` after each `evict` or `prefetch`, into its output file in the test's directory (`/tmp/vmemd-WORKLOAD-*/workerN.out`). The tracer below counts these lines to show hints over time.

## Memory over time (`make trace-all`)

These targets record how memory use changes while the four-process tests run, and plot the runs of all backends on one chart per workload. Two scripts at the top of the repository do the work, and they also work for programs outside these tests.

`scripts/memtrace.py` runs a command and writes one CSV. Every 0.1 s (`--interval` changes it) it records the time since the command started, the GPU's used memory, the machine's used DRAM, and one column per `--count`. VRAM comes from one `nvidia-smi` in loop mode and covers the whole GPU. DRAM is `MemTotal - MemAvailable` from `/proc/meminfo`, which covers the whole machine and includes pinned memory, since pinned memory does not show up in any process's size. A `--count LABEL=GLOB:REGEX` counts the lines matching `REGEX` in the files matching `GLOB` that were written after the trace started. It reads the glob again at every sample, so the files can appear after the command starts. The script exits with the command's exit code.

`scripts/memplot.py` draws one or more of these CSVs on one chart. Each run gets its own color. The left axis shows VRAM (solid) and DRAM (dashed) in MiB above their values at the start, and a faint dotted line in the same color marks each peak. `--right COLUMN` draws one count column on the right axis as a dash-dot line, and `--right-for` picks which runs get it.

In `libvmem/`, `make trace-gemm` and `make trace-image` trace one backend (`BACKEND=vmem`, `vmem-local`, `cuda` or `uvm`) and write `libvmem/analysis/WORKLOAD-BACKEND.csv`. They skip the solo pass, so the trace shows only the four workers running together, and they count the `hint ok` and `hint refused` lines as `hints accepted` and `hints refused`. `make trace-all` runs both workloads on all four backends and then `make plot-traces`, which draws `libvmem/analysis/gemm.png` and `libvmem/analysis/image.png` with the accepted hints of `vmem local` and `vmem + vmemd` on the right axis. The plots leave `cuda` out, because it runs out of memory in both workloads; its CSVs are still written. A run that fails, such as `cuda` running out of memory, still leaves its CSV. The plots and what they show are in [RESULTS.md](RESULTS.md).

To trace a larger program the same way:

```
scripts/memtrace.py -o analysis/myapp.csv --count "hints refused=/tmp/vmemd.log:move of .* refused" -- ./myapp args
scripts/memplot.py -o analysis/myapp.png --right "hints refused" myapp=analysis/myapp.csv
```

`memtrace.py` needs only `nvidia-smi`; `memplot.py` needs `matplotlib`.

## Known limitations

Phase 6 of `test-pair` assumes nothing else uses the GPU, because it expects `c`'s 256 MiB chunk to fail on the device right after `a` filled it. Another program that allocates or frees VRAM in between can change that. In the image test with six images per worker, whether the `cuda` baseline fits depends on how much the four workers overlap in time, so its outcome changes from run to run. The image test takes 1 to 5 s, so at the default interval a trace of it has only 12 to 54 samples, and the tracer can miss peaks shorter than one sample. VRAM and DRAM in the traces cover the whole GPU and the whole machine, so other programs show up in them; run the traces with nothing else on the GPU and little else on the machine.
