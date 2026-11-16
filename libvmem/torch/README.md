# torch-vmem

torch-vmem makes `libvmem` the CUDA allocator of an unmodified PyTorch program. This README tells how it was built, in three phases. Each phase solves one problem, adds one piece of code, and has one test that proves it works. Read them in order; each builds on the last.

The plan, the hypotheses and the design choices are in [`notes/pytorch-plan.md`](../../notes/pytorch-plan.md). Tracker items: 6B (allocator) and 6C (hints).

## Why we are doing this

`libvmem` showed that VMM chunks can live in VRAM or in pinned host memory, and that a chunk on host stays usable: kernels read it over PCIe, slower but correct. What `libvmem` could not know is which memory a kernel will touch next. A general CUDA program never says.

PyTorch does say. Before each op runs, PyTorch knows on the host which tensors the op reads and writes. So if PyTorch allocates from `libvmem`, we can move tensors to host after their last use and back before their next one. That gives us three things to test against plain PyTorch:

- H1: VRAM plus host memory lets a job run that would otherwise run out of memory.
- H2: smart evict and prefetch makes it faster, including faster than UVM given the same hints.
- H3: with several processes, `vmemd` coordinating all of them beats each acting alone.

We built the machinery in three phases. None of them contains a real policy yet. That is the next step.

## Setup

You need the venv with PyTorch 2.14 built for CUDA 13.0 in `torch/.venv`:

```
cd libvmem
uv venv torch/.venv --python 3.12
uv pip install --python torch/.venv/bin/python torch numpy --index-url https://download.pytorch.org/whl/cu130
make torch          # builds bin/libtorchvmem.so from torch/alloc.cpp
make test-torch     # 4 tests x 3 backends, about a minute
```

`VMEM_BACKEND=vmem|uvm|cuda` picks the backend (default `vmem`). `VMEM_DAEMON` sets the `vmemd` socket (default `/tmp/vmemd.sock`). With no broker running, `vmem` works alone in local mode.

## Phase 1: PyTorch runs on libvmem

**Problem.** PyTorch must allocate every CUDA tensor from `libvmem`, without changes to PyTorch or to the model code.

**What we added.** PyTorch lets you swap its allocator for two C functions through `CUDAPluggableAllocator`. We wrote them in `alloc.cpp` and load them in `torchvmem.install()`:

```python
import torchvmem as tv
tv.install()                 # before the first CUDA tensor
model = MyModel().cuda()     # every CUDA tensor now comes from libvmem
```

The allocator reuses the `libvmem` backends (`../src/backends/`) as they are, but not the older `../src/libvmem.cuh`. That one has no caching and no idea of streams.

- A tensor larger than 1 MiB gets its own backend chunk, rounded up to 2 MiB. Only these chunks can move later.
- A tensor of 1 MiB or less gets a power-of-two slot inside a shared 2 MiB chunk. Slots never move.
- Free never gives memory back. The block goes to a cache, and the next tensor of that size reuses it. Without the cache, every tensor would cost a `cuMemCreate` call (0.2 to 0.6 ms).
- When the backend runs out of memory, the allocator destroys the cached large chunks and tries once more.
- One mutex guards everything, because the autograd thread allocates too.

**Test: `check.py`.** A small MLP trains for 5 steps on the GPU and on the CPU. The losses must match (0.517937 vs. 0.517936), and a second run must not grow the live bytes.

## Phase 2: reuse without stalls

**Problem.** PyTorch frees a tensor on the host as soon as its last reference dies. Kernels that use it may still be queued on the GPU. Handing that memory to a new tensor too early corrupts data. Waiting for the whole GPU on every free is safe but drains the GPU queue each time.

**What we added.** CUDA events, one per block.

- At free, we record an event on the tensor's stream. The event completes when everything queued so far on that stream is done.
- Reuse on the same stream needs no wait, because kernels on one stream run in order. The allocator prefers such a block.
- Reuse on another stream makes that stream wait for the event on the GPU (`cudaStreamWaitEvent`). The host never blocks.
- Destroying a cached chunk waits for that chunk's event only, not for the whole device.

We also changed the `vmem` and `uvm` backends to copy on their own non-blocking stream, so a move does not wait for unrelated work on PyTorch's default stream.

**Test: `test_streams.py`.** A side stream queues 1000 slow adds on a tensor and frees it. The main stream gets the same memory and fills it with -1. Without the event wait, the adds land in the new tensor and the test fails. We checked that it does fail with the wait removed.

## Phase 3: a harness for policies

**Problem.** A policy needs to know, for every op, which tensors it uses, and it needs a safe moment to move a tensor. Moving a chunk while a queued kernel still uses it crashes that kernel or loses its writes.

**What we added.** `torchvmem.Harness`, a `TorchDispatchMode` that sees every op, forward, backward and optimizer alike. For each op it does five things:

1. It finds the addresses the op reads and writes. It reuses `ArgumentHandler` from PyTorch's CUDA sanitizer, which reads the op schema.
2. It calls `policy.before(h, op)`, where a policy prefetches.
3. It runs the op.
4. It calls `vmem_touch`. This records a last-use event on every block the op used, and updates the block's `last_op` and `uses`.
5. It calls `policy.after(h, op)`, where a policy evicts.

`evict` and `prefetch` wait for the block's last-use event, then move the chunk. If the harness never touched the block, they sync the whole device instead. Moves are blocking: the call returns when the chunk is at its target.

The recorded state lives in two places:

- The allocator keeps, per block, the last-use event, `last_op`, `uses` and the location. These must be exact, and the moves need the event.
- The harness keeps the op trace in Python: `h.trace` for the current iteration, `h.prev` for the last finished one. Training repeats the same ops every iteration, so a policy can look ahead in `h.prev`.

A policy is a class with three hooks:

```python
class MyPolicy(tv.Policy):
    def before(self, h, op):   # op.reads, op.writes: sets of addresses
        ...
    def after(self, h, op):    # the op's kernels are queued; evicting waits for them
        ...
    def on_step(self, h):      # end of iteration; h.trace holds its ops
        ...

h = tv.Harness(MyPolicy())
with h:
    for batch in data:
        train_step(batch)
        h.step()               # once per iteration
```

`tv.evict(p)` and `tv.prefetch(p)` take a tensor or an address. They return `tv.OK`, `tv.NO_MEMORY` (no room, or `vmemd` refused), `tv.UNSUPPORTED` (a small tensor, an unknown address, or the `cuda` backend) or `tv.FATAL`. `tv.block(p)` returns the block's base, size, location, `last_op` and `uses`. `tv.info()` returns the live, cached, device and host bytes.

**Tests.**

- `test_moves.py` evicts a 16 MiB tensor by hand, computes on it while it sits on host, prefetches it back, and checks the data. Small tensors must refuse to move.
- `test_harness.py` trains with the worst possible policy, which evicts every large tensor after each op and prefetches before each op. The loss must still match the CPU's. It also checks that the harness saw forward, backward and Adam ops, and that blocks carry `last_op` and `uses`.

## Reference

### Files

| File | Phase | What it is |
|---|---|---|
| `alloc.cpp` | 1-3 | The allocator, built into `bin/libtorchvmem.so` |
| `torchvmem.py` | 1, 3 | `install()`, the direct calls, and `Harness`, `Policy`, `Op` |
| `check.py` | 1 | Trains to the CPU's loss, no leak |
| `test_streams.py` | 2 | Cross-stream reuse waits |
| `test_moves.py` | 3 | Evict, compute on host, prefetch |
| `test_harness.py` | 3 | Worst-case policy still trains correctly |

### Backends

| `VMEM_BACKEND` | Memory | Moves |
|---|---|---|
| `cuda` | `cudaMalloc` | none |
| `uvm` | `cudaMallocManaged` | `cudaMemPrefetchAsync`; the driver also moves pages on faults |
| `vmem` | VMM chunks in VRAM or pinned host memory, placed by `vmemd` if one runs | backing swap; a host chunk stays mapped and kernels read it over PCIe |

### Limits

- `tensor.record_stream()` is ignored. PyTorch exposes that hook only in C++, and we decided against a C++ extension. While the harness runs, a touch on a second stream covers the case.
- A policy must not evict a tensor that a later, already-queued op still uses. The harness only knows about ops it has seen.
- Small-tensor chunks are never destroyed, and small tensors never move.
- A cached chunk keeps its location, so a new tensor can get a host chunk from the cache and stay there.
- `torch.cuda.memory_allocated()` and the other memory stats fail with a pluggable allocator. Use `tv.info()` or `scripts/memtrace.py`.
- Every large tensor is rounded up to 2 MiB.
- The harness costs a Python call per op. We have not measured it.
- Only device 0, eager mode. `torch.compile`, CUDA graphs and NCCL are untested.

## Next steps

Implementation:

- [ ] Write a first real policy (6C): evict a block after its last use in the iteration, and prefetch it a few ops before its next use, both read from `h.prev`.
- [ ] Prefer device blocks when a new tensor takes a block from the cache.
- [ ] Run the same policy on the `uvm` backend as the baseline (6C).
- [ ] Measure the per-op harness cost. If it is large, touch only blocks larger than 1 MiB, or move the hook to C++.
- [ ] Later (6D): non-blocking moves that overlap with compute.

Testing:

- [ ] Run the tests with `vmemd` running (`make daemon`, then `bin/vmemd --config daemon/vmemd.toml`), including a refused move.
- [ ] Add a model larger than VRAM: native PyTorch runs out of memory, `vmem` spills to host and finishes.
- [ ] Test running out of memory with a full cache: the allocator must trim and retry.

Benchmarking (6B):

- [ ] Pick one training script that fits in VRAM and one that does not.
- [ ] Compare native PyTorch, native with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and torch-vmem on `cuda`, `uvm`, `vmem`, and `vmem` with `vmemd`.
- [ ] Record iteration time, peak VRAM, peak pinned DRAM and the 2 MiB rounding waste. Run each case 5 times and report the median and range.
