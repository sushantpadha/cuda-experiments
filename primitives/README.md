# libvmem v0 primitives

**Status: partial, work in progress.** Tracker 1A. `src/vmem.cuh`: a `vmem::Manager` wrapping the CUDA VMM driver API in blocking, error-checked calls. `bin/runner` runs `.vm` action files on one Manager.

## Layout

```
src/vmem.cuh        the Manager (header only)
src/pattern.cuh     fill/check kernels: word i = k * (i + 1); shared by runner and tests
src/runner/         the action runner, one file per part (see src/runner/main.cu)
actions/            .vm action files: input for the runner
tests/              test programs: testN.cu (C++ on the Manager), fuzz.py (runner vs. a model)
bin/  logs/         build output, runner logs (not tracked)
```

**Tests vs. actions.** A test checks something and passes or fails on its own: `tests/testN.cu`, `tests/fuzz.py`. An action file is a script of Manager calls the runner executes and logs: `actions/*.vm`. `actions/test1.vm` and `actions/test2.vm` replay `tests/test1.cu` and `tests/test2.cu` as actions.

## Build and run

```
make                    # bin/runner
make test N=1           # build and run tests/test1.cu
make test N=2 ARGS=-d   # tests/test2.cu with state dumps
make test N=3 ARGS="12 24"
make run N=1            # bin/runner on actions/test1.vm; ARGS passes runner flags
make fuzz               # tests/fuzz.py; ARGS="-n 200" for more programs
make debug              # -g -G -DDEBUG build
```

| Test | Checks |
|---|---|
| `test1.cu` | basic workflow: reserve, create, map, fill, remap to host and back, check, tear down |
| `test2.cu` | every call, data across remaps, error paths |
| `test3.cu` | timing: remap a MiB to host vs. `cudaMemcpyAsync` b MiB |
| `fuzz.py` | random `.vm` programs; a model of the Manager's rules predicts which lines fail; runner must agree on every line |

## v0 subgoals (tracker 1A)

- [x] Primitives in `src/vmem.cuh`: reserve, create (device or pinned host), map, unmap, remap, release, free; residency queries; chunk tokens
- [x] Tests: `tests/test1.cu` (basic), `tests/test2.cu` (every call, error paths), `tests/fuzz.py`
- [~] `tests/test3.cu`: first timing test (remap vs. `cudaMemcpyAsync` to host); under review
- [ ] Review `vmem.cuh` (API, remap path, destructor, error handling)
- [x] Action runner (1Ai): `bin/runner` checks a `.vm` file, then runs it on one Manager
- [ ] Parallel runs (1Aii): many runner processes at once; check correctness and basic speed

Not in v0: threads, async remap, hooks, policies.

### Notes to self: Toward a daemon model

What a later shim + daemon design will need on top of this:

- one `Manager` per tenant process: only the owning process can map its own addresses
- a command channel so the daemon can ask a process to remap or release a chunk (tokens are plain integers, so they travel over IPC)
- thread safety: commands arrive on another thread
- knowing when a chunk is idle, so a remap never races a running kernel -- HOW IS THIS POSSIBLE?
- a host-memory budget across processes -- Note that MPS limits can cover device memory but not host
- handle export (`cuMemExportToShareableHandle`) if chunks are shared between processes
- pre-created/set access'ed chunks sitting on device and host (minor though since remap cost ~ mem copy cost)

## Open issues to review

- `tests/test3.cu` calls `remap` right after launching `fill` on a non-blocking stream. `remap`'s copy is ordered only after the legacy default stream, so it can race the kernel; sync the stream first.
- `remap` is blocking; the copy could run async, with the swap done once it lands.

## Manager API

Per-process: one `Manager` per process, owning only that process's memory. Not a shared allocator, not a daemon.

| Call | Does | Returns |
|---|---|---|
| `Manager(Options)` | `cuInit`, primary context, checks VMM and host-NUMA support, granularity | |
| `reserve(size)` | reserve a virtual range | base VA |
| `create(size, Loc::Device \| Loc::Host)` | physical memory on the GPU or in pinned host memory | `ChunkToken` |
| `map(tok, va)` | map inside a reservation, grant device read/write | `va` |
| `unmap(tok)` | unmap from its stored address | that address |
| `remap(tok, Loc)` | move a mapped chunk, same VA, data copied; prints and does nothing if already there | its address |
| `release(tok)` | free physical memory (must be unmapped) | chunk size |
| `free(va)` | free a reservation (nothing mapped inside) | its size |
| `va(tok)`, `ptr<T>(tok)` | mapped address, stable across remap | `CUdeviceptr`, `T*` |
| `loc(tok)`, `on_device(tok)` | residency now | `Loc`, `bool` |
| `info(tok)` | id, size, location, address | `ChunkInfo` |
| `print_state()` | reservations, chunks, free VRAM | |

- `ChunkToken`: copyable value; never reused, so a stale or foreign token throws.
- Sizes round up to the granularity (2 MiB here). Failures throw `vmem::Error` (`.code` = `CUresult` if a CUDA call failed).
- `Options{device, host_numa, verbose, debug, out}`: GPU (default 0); NUMA node for host chunks (default -1 = closest to the GPU, else 0); one line per call; state dump after each call; `out` gets every printed line instead of stderr.
- Destructor unmaps, releases and frees what is left.
- Rules: single-threaded (mutex later); never `unmap` or `remap` a chunk a kernel may touch (crash, no fault handler); `remap`'s copy is ordered after the legacy default stream only.

## Action runner

`bin/runner [-v] [-d] [--stable] [--pid] [--parse] [-o logdir] file.vm`

Checks the whole file first (syntax, unknown calls, argument count, undefined variables, wrong kinds); runs nothing if anything is wrong. Then runs each line on one Manager, logs it, continues after errors.

```python
# comments and blank lines are ignored; names are [A-Za-z_][A-Za-z0-9_]*
r = reserve(64M)              # sizes: bytes, or K M G (binary)
c = create(32M, device)       # device | host
map(c, r + 32M)               # address = reservation + offset (map only)
remap(c, host)
v = unmap(c)                  # assigning a result is optional
release(c)
free(r)
va(c)                         # also: loc(c), on_device(c), info(c), granularity(), print_state()
fill(c)                       # word i of chunk #id = (id + 1) * (i + 1), uint32
check(c)                      # the line fails if any word differs
sleep(10ms)                   # us | ms | s
```

- Calls return what the Manager returns (address, chunk, size). A variable takes the kind of its last assignment.
- `fill` and `check` synchronize: each line is done before the next starts.
- Output: header (file, start time, options, program size, log paths) and a `PASS`/`FAIL` result, on the terminal and in the log.
- Log `logs/<name>.log`: one block per line: line, statement, `ok`/`ERROR`/`SKIP`, time, result or error, then the Manager's output indented. `init`/`end` blocks: Manager constructor and destructor.
- Broken CUDA context (e.g. illegal address in a kernel): reported on that line, the rest is skipped.
- Exit code: 0 clean, 1 runtime errors, 2 bad file or options.

| Flag | Effect |
|---|---|
| `-v` | print the log while running |
| `-d` | also `logs/<name>.debug.log`, with `print_state()` after every line |
| `--stable` | addresses become `va#1`, `va#2`, …; times and free-VRAM numbers dropped: two runs diff cleanly |
| `--pid` | logs named `<name>.<pid>.log`, for many runners on one file |
| `--parse` | check the file only |
| `-o dir` | log folder (default `logs`) |

`actions/test2.vm` ends with exactly 7 errors: they replay `test2.cu`'s expected failures, marked `# error:`.

## Fuzz test

`tests/fuzz.py` (AI-generated) writes random `.vm` programs, predicts with a small model of the Manager's rules which lines must fail (mapped or not, released, inside a reservation, overlap, data filled), runs each through the runner with `--stable`, and compares ok/ERROR per line. Exit 0 when they all agree.

```
make fuzz                                    # 50 programs x 60 lines
tests/fuzz.py -n 300 -l 120 --seed 1000      # more, longer, other seeds
```

Programs and logs go to `logs/fuzz/`. A disagreement prints the seed, line and statement; rerun that seed to look at `logs/fuzz/fuzz<seed>.vm` and its log. Addresses past a reservation's end belong to the next one (reservations are contiguous), so the generator keeps offsets inside the reservation.
