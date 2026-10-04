# CUDA streams vs. graphs, and how Nixie handles them

2026-09-27. Tags: **[docs]** CUDA Programming Guide; **[src]** Nixie code at `eebf583` (see `nixie-summary/source-notes.md`); **[inferred]** follows from code and docs, not run.

## Streams

- Ordered queue of GPU work. The CPU submits every item as its own API call (`cudaLaunchKernel`, `cudaMemcpyAsync`, ...). Calls return at once; the GPU runs them later, in order within the stream. Separate streams may run concurrently. **[docs]**
- Cost: CPU launch overhead per item. An LLM decode step can be hundreds of small kernels, so hundreds of calls per token.

## Graphs

- Record once, replay many times. **[docs]**
  1. Capture: `cudaStreamBeginCapture`, run the stream code (work is recorded, not executed), `cudaStreamEndCapture` returns a graph. Or build nodes explicitly.
  2. Instantiate: `cudaGraphInstantiate` gives an executable graph.
  3. Replay: `cudaGraphLaunch` submits the whole DAG in one call.
- Kernel arguments, including device pointers, are frozen at capture. Memory must stay at the same address for replays to be valid.
- Capture is fragile. In the default global capture mode some calls from any thread (for example device-wide synchronisation, `cudaMalloc`) are not allowed and invalidate the capture; the app sees an error at end of capture. Exact list: Programming Guide, stream capture section (not yet opened).
- Graphs can own memory through memory nodes (`cudaGraphAddMemAllocNode`, or `cudaMallocAsync` captured into a graph).
- llama.cpp and SGLang use graphs for decode; SGLang captures many at startup.

## Nixie

What fits:
- Stable virtual addresses (reserve once, remap physical memory behind them) keep frozen graph pointers valid. The key property. **[src]**
- Gate is per process, not per stream. Disable blocks every thread and `cuCtxSynchronize()` drains all streams, so stream ordering is preserved. Nixie's own copy streams run only while the app is paused. **[src]**
- One `cudaGraphLaunch` passes the gate once. Idle timeout for graphs 200 ms vs. 100 ms for kernels. **[src]**
- Paper evaluates SGLang and llama.cpp successfully (§7).

Problems:
1. Capture-begin hook exported as `cudaStreamCaptureBegin` (`intercept_launch.rs:62`); the API is `cudaStreamBeginCapture`. Apps never call the misnamed export, so the capture flag is never set and the wait in `set_allow_running` (`schedule/mod.rs:81`) never triggers. The §4 "CUDA graph compatibility" behaviour is not active in this commit. **[src]**
2. A Disable during capture calls `cuCtxSynchronize()`, disallowed during global-mode capture, so the capture fails. Likely rare: captures happen at startup and equal-priority preemption waits out an 8 s cooldown; a higher-priority arrival can still hit it. **[inferred]**
3. Even if renamed: one global flag for all streams; two overlapping captures clear it early. **[src]**
4. Nixie's `cudaMallocAsync` synchronises the stream; on a capturing stream that is illegal, so stream-ordered allocation inside a capture breaks. **[inferred]**
5. Graph-owned memory never passes through `cudaMalloc`: invisible to Nixie, stays in VRAM while paused. **[inferred]**

## To do

- Test for problem 1 and 2: a program that captures a long graph under `nixie run` while a second, higher-priority app requests the GPU.
- Read the Programming Guide stream-capture rules and quote the exact prohibited calls.

## Relevance to libvmem

Any remapping library must keep VAs fixed (graphs), must not issue illegal calls during capture, and must see graph-owned and driver-API memory or document that it cannot.
