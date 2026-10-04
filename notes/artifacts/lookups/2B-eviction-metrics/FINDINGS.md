# 2B: signals for an eviction policy, without hardware access bits

2026-10-04. Tracker 2B. Same machine as 2A. Code and commands: `experiments/lookups/2B/` (`argshim.cpp`, `cupti_shim.cpp`, `argapp.cu`, `signals.cu`, `README.md`). 1 to 3 runs each: preliminary. Independent re-check in `../REVIEW.md`; corrections folded in.

Tags: **[run]** measured here; **[src]** NVIDIA open kernel modules 580.178.04 (`kernel-open/nvidia-uvm/`) or CUDA 13.0 headers; **[inferred]** follows from the above, not run.

## Answer

Two cheap, transparent signals work; hardware ones are out of reach.

| Signal | Works? | Cost | Gives |
|---|---|---|---|
| Kernel-argument scan, CUPTI-injected | **yes** | at most about 0.2 us per launch, inside run-to-run noise (baseline 1.69 to 1.82 us/launch, CUPTI 1.87 to 2.05); the scan itself 31 to 34 ns on a 0-parameter kernel | which allocations each launch may touch, incl. cuBLAS; per-allocation use count and last use |
| Kernel-argument scan, `LD_PRELOAD` on exported symbols | yes for app kernels, **missed cuBLAS** | about 30 ns per launch | same, app kernels only |
| Kernel time vs. fraction demoted | **yes** | free (events) | monotone "pain" per tenant: streaming pass 6.5 to 48 to 53 ms as 0 to 16 of 16 chunks go to host (two runs) |
| GPU-side block hashing | **yes** | 188 GB/s (1 GiB in 5.7 ms) vs. 13.1 GB/s copy-out | which 2 MiB blocks changed (dirty), exact on a 1-byte change |
| Hardware access counters | **no** (not usable from userspace) | | |
| CUPTI address-level sampling | **no** for non-UVM memory | | |

## Details

**Kernel-argument scan.** Track allocations; at each launch, read the parameter layout (`cuFuncGetParamInfo`, or `cuKernelGetParamInfo` when the handle is a `CUkernel`) and check every 8-byte word of every parameter against the allocation map. Test app ground truth vs. detected **[run]**:

| Launch | Truth | Detected |
|---|---|---|
| `k_direct(A,B)` | {A,B} | {A,B} |
| `k_struct(Pair{C,n,D})` (pointers inside a by-value struct) | {C,D} | {C,D} |
| `k_interior(A + 1 MiB)` | {A} | {A} |
| `k_indirect(T)`, T holds a pointer to F | {T,F} | {T}: F missed, as expected |
| `cublasSgemm(A,B,C)` | {A,B,C} | `LD_PRELOAD`: launch not seen at all. CUPTI: `ampere_sgemm_128x64_nn` -> {A,B,C, cuBLAS workspace} |

- **CUDA 13 changed the launch entry point.** nvcc 13 stubs call `__cudaLaunchKernel(cudaKernel_t, ...)` (`crt/host_runtime.h:140`, `crt/device_functions.h:2932`), not `cudaLaunchKernel`; `nm -D argapp` shows only `__cudaLaunchKernel`. The first shim version, hooking only `cudaLaunchKernel`, counted zero launches **[run, src]** (that version is not kept). Nixie (`eebf583`) hooks `cudaLaunchKernel` and `cudaGraphLaunch` (`intercept_launch.rs:26,51`) but not `__cudaLaunchKernel`, so it would not gate `<<<>>>` launches of apps built with CUDA 13 **[inferred]**; its paper used CUDA 12.9. Any runtime-symbol hook also fails on binaries linked with static cudart (nvcc's default).
- **This `LD_PRELOAD` shim misses library launches.** cuBLAS links cudart statically and resolves driver calls through entry points, so its launch bypassed the exported `cudaLaunchKernel`, `__cudaLaunchKernel` and `cuLaunchKernel` symbols **[run]**. An `LD_PRELOAD` that interposes `cuGetProcAddress`/`dlsym` (as HAMi-core does) could see them; not tried.
- **CUPTI injection sees driver-level launches, library ones included.** `CUDA_INJECTION64_PATH=cupti_shim.so`, driver-API callbacks. No relink, no `LD_PRELOAD`. Caught all 1005 launches including cuBLAS **[run]**. Coverage of this probe: launches via `cuLaunchKernel(Ex)(_ptsz)` only (not cooperative launches, `cuGraphLaunch`, host functions); allocations via `cuMemAlloc_v2`, `cuMemAllocAsync`, `cuMemMap` only (not pools, managed, pitch; frees via `cuMemFree_v2`, `cuMemUnmap` only). Extending the callback list is mechanical. A v1 transparent hook could use the same mechanism for launch gating.
- Limits: pointers stored in device memory (pointer arrays, linked structures) are invisible; "may touch" is not "did touch"; CUDA graph launches would need node inspection at instantiate (not tested).

**Kernel time as a pain signal.** 512 MiB buffer in 16 x 32 MiB VMM chunks, k chunks backed by `HOST_NUMA` **[run]**:

| Chunks on host | 0 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| Streaming read-modify-write, ms | 6.5 | 17.4 | 26.1 | 40.2 | 48.0 |
| Same data touched 8 times, ms | 52 | 108 | 220 | 329 | 543 |

Roughly linear in the demoted fraction (re-check, streaming: 6.5, 16.1, 26.5, 37.2, 52.6 ms; the 16/16 values vary by 10 to 20% between runs). So per-tenant kernel time measured with events tracks how much demotion hurts, which supports marginal-utility allocation of VRAM. Caveat: under MPS a tenant's kernel time also depends on its co-tenants, so the signal is confounded.

The 8-pass row is **not** a reuse test: 8 streaming passes over 512 MiB never fit the 24 MB L2, and both rows show the same 8x host/device ratio (re-check: 8.1x and 8.3x). An earlier claim that "reuse costs more on host" was wrong. What the numbers do give is a promotion break-even: one streaming pass over 512 MiB costs about 41 to 46 ms extra on host, and copying 512 MiB to VRAM at 13 GB/s costs about 40 ms. So data touched more than about once per residency period is worth promoting, which makes per-allocation use counts (from the scan) the promotion signal.

**Hashing.** One CTA per 2 MiB block, order-independent mixed sum: 1 GiB in 5.71 ms (187.9 GB/s, at VRAM bandwidth). After writing one byte in blocks 137 and 400, exactly blocks 137 and 400 changed hash **[run]**. Copying the same 1 GiB to pinned host takes 81.9 ms, so checking dirtiness costs about 7% of copying. The mix (odd multiply, xor-shift) is a bijection per word, so any single-word change changes the sum; multi-word changes can collide with low probability (64-bit). Tested on uniform data in `cudaMalloc` memory only, not VMM.

**Hardware access counters: not usable.** **[src]**
- Consumed only inside `nvidia-uvm`. Support depends on `accessCntrBufferCount` reported by the closed RM (`uvm_gpu.c:1435`); disabled under vGPU or confidential computing (`uvm_hal.c:914`).
- Migrations from counters are off by default on x86 without ATS: `uvm_perf_access_counter_migration_enable = -1` and the policy returns false unless ATS (`uvm_gpu_access_counters.c:148-156`). This machine reads `-1`.
- Notifications for addresses outside UVM-managed ranges go to the HMM path or "tools" events (`uvm_gpu_access_counters.c:1538-1600`); VMM ranges are not UVM ranges, so counters cannot be pointed at our memory.
- Whether this GPU reports any counter buffers needs `uvm_enable_debug_procfs=1` (module reload, root). Not checked. The tools access-counter events are test-only (`UvmEventTypeTestAccessCounter`, behind `uvm_enable_builtin_tests=1`, `uvm_test.c:245`; re-check).

**CUPTI address data: UVM only.** CUPTI activity records carry addresses only for allocations and UVM fault/transfer events; PC sampling gives PCs and stall reasons, SASS metrics give per-instruction counts **[src]** (`cupti_activity.h`, `cupti_pcsampling.h`, `cupti_sass_metrics.h`). Nothing reports which VMM pages a kernel touched. For offline ground truth: the Compute Sanitizer API is already installed and gives a per-access address callback (`SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS`, `/usr/local/cuda/compute-sanitizer/include/sanitizer_patching.h`); NVBit is the other option (not installed). Neither tried.

## What this means for libvmem

- Recency and frequency per allocation: CUPTI launch scan (cheap, sees libraries). Pair with stream events for "last use finished".
- Cost to evict: block hashes (clean blocks drop for free).
- Benefit of VRAM per tenant: kernel-time slowdown vs. demoted fraction.
- What to promote: use count per allocation from the scan; break-even is about one pass per residency period at these bandwidths.
- The indirect-pointer blind spot is exactly where the host fallback matters: a missed allocation gets slow, not wrong (and under MPS, wrong would kill co-tenants, see 2A).

## Open

- Real workloads: PyTorch is not installed on this machine, so no framework run yet. Rodinia or cuDNN-based apps next.
- CUDA graphs: read kernel nodes at `cuGraphInstantiate`.
- Overhead with realistic kernels (about 20 parameters, as sgemm) and many launching threads (the shims take one global lock).
- Prototype "last use finished" via stream events.
