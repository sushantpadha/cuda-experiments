# 2A: compute sharing across processes, and VMM under it

2026-10-04. Tracker 2A. Machine: RTX 4050 Laptop (20 SMs, 6 GB), driver 580.178.04, CUDA 13.0. Code and exact commands: `experiments/lookups/2A/` (`spatial.cu`, `README.md`). Most numbers are 1 to 3 runs on this machine: preliminary, not report facts. An independent re-check is in `../REVIEW.md`; corrections from it are folded in below.

Tags: **[run]** measured here; **[docs]** NVIDIA docs (CUDA Programming Guide 13.3 §4.6, MPS docs); **[src]** Nixie code at `eebf583`.

## Answer

- **Without MPS, kernels from different processes do not overlap; they time-slice.** Two processes, same fixed-work kernel on 4 blocks (16 of 20 SMs idle): 452 ms alone, 915 ms each when run together (3 trials: 915 to 933 ms). Control: 8 blocks in one process take 444 ms, so the doubling is not a lack of SMs. **[run]**
- **With MPS, they run concurrently.** Same test: 2 and 3 processes each finish in 444 to 445 ms, the same as alone (3 trials). **[run]**
- **Green contexts partition SMs inside a process; across processes they need MPS.** One process, two green contexts of 10 SMs each: disjoint SM sets ({0,1,12..19} vs {2..11}), both concurrent. Two processes with disjoint green contexts and no MPS: still time-sliced (1310 ms vs 627 ms alone). **[run]**
- **Green contexts plus MPS give isolated SM partitions across processes.** Two MPS clients, 40 blocks each, 10-SM green contexts: disjoint groups 630 / 630 ms (2 trials; 638 in the re-check), same group 658 to 670 / 1182 to 1206 ms. Disjoint clients do not slow each other; clients on the same group do. **[run]** This is the strongest SM-isolation option on CUDA 13.0 (MPS static partitioning needs 13.1).
- **VMM works fully under MPS.** `cuMemCreate` (device and `HOST_NUMA`), map, set access, kernel read/write, remap at the same VA, `cuMemExportToShareableHandle` (POSIX fd) and import in another MPS client: all pass, all data checks 0 errors. Numbers match the no-MPS runs. **[run]** The MPS docs list no VMM restriction (Linux). **[docs]**
- **MPS per-client limits apply to VMM device memory, not to host memory.** `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=128M` makes a 256 MiB device `cuMemCreate` return `CUDA_ERROR_OUT_OF_MEMORY`; 1G passes. Under the same 128M cap, a 256 MiB `HOST_NUMA` `cuMemCreate` succeeds. `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25` gives about 4 SMs (1503 ms vs 470 ms; a 4-SM green context gives 1494 ms for the same kernel, re-check). **[run]** So MPS gives per-tenant VRAM caps and SM shares for free; host fallback memory needs libvmem's own budget.
- **Harsh negative: under MPS, one client's fault kills the others, busy or idle.** Client B runs a 3 s kernel; client A touches a reserved but unmapped VMM range: A gets `cudaErrorIllegalAddress` and **B's kernel also fails with `cudaErrorIllegalAddress`** (3 of 3 trials). A client that is idle when A faults gets the same error on its next launch (`./spatial idlecheck`). The server log shows the device context destroyed and recreated; a new client works afterwards. Without MPS, the same overlap leaves B unharmed (overlap confirmed with host timestamps in the re-check). Xid 31 (MMU fault) logged, no reboot needed. **[run]** Docs agree: "a fatal fault from one client may bring down a different user's client". **[docs]**

## Fallback and remap numbers (supporting 1A design)

| Measurement | Value **[run]** |
|---|---|
| Kernel seq. read, device-backed VMM | 187 GB/s |
| Kernel seq. read, `HOST_NUMA`-backed VMM (fallback) | 13.4 GB/s (about 7% of device) |
| Kernel random 4 B reads, device / host-backed | 1860 / 128 M reads/s (about 15x slower) |
| Kernel write to host-backed, then verify | 0 errors |
| Remap 256 MiB device to host at same VA: copy | 20 ms (13.2 GB/s) |
| Remap: unmap + release + map + set access | 2.7 ms |
| Two `./spatial vmm` at once under MPS | copy 5.2 to 6.6 GB/s each, host reads 6.7 to 8.9 GB/s each, depending on overlap (shared PCIe link) |
| VRAM per bare driver context, no runtime modules loaded (no MPS / MPS client) | about 87 / 80 MiB device-used growth; MPS server adds about 30 MiB once. Real apps load more |
| Allocation granularity, device and `HOST_NUMA` | 2 MiB |

- **Remap does not wait for unrelated kernels.** In one process, a 450 ms kernel runs on buffer X while buffer Y (64 MiB) is remapped device to host: create 8.6 ms, copy 5.2 ms, unmap 0.05 ms, map 1.5 ms; the kernel was still running when the remap finished, data verified. So per-buffer remap needs no context-wide sync. (Nixie calls `cuCtxSynchronize` at `schedule/mod.rs:116` when it pauses a whole app, a different operation.) Safety then rests on knowing the kernel does not touch Y. Single process only; cross-process remap under MPS not tested. **[run]**
- **Remap under a kernel that reads the range crashes.** A kernel reads a 2 MiB buffer for about 1 s; the host unmaps it and maps a host handle at the same VA 200 ms in: `cudaErrorIllegalAddress`, 2 of 2 runs. `cuMemMap` over an already-mapped range returns `CUDA_ERROR_INVALID_VALUE` (re-check scratch test; not in `spatial.cu`), so there is no atomic swap: remap is only safe at a point where no kernel can touch the buffer. **[run]** (`./spatial gaprace`, 3 of 3 in the re-check)
- `%smid` under MPS looks virtualised (two MPS clients with disjoint green contexts both report SMs 0..9 yet run at full single-green speed; a 25% client reports SMs 0..3). Do not use `%smid` to prove SM isolation under MPS. **[run, inferred]**

## What this means for libvmem

1. Spatial compute sharing across processes needs MPS. Green contexts alone do not give it; green contexts inside MPS clients do give isolated SM partitions.
2. MPS makes the host fallback a correctness requirement, not an optimisation: an evicted range left unmapped crashes every tenant on the GPU, running or idle. Evicted ranges must always stay mapped (to `HOST_NUMA`), and remaps must never leave a gap while any kernel may touch the range.
3. MPS already supplies per-client VRAM caps and SM caps; libvmem's policy can set these instead of re-implementing quotas. Caveats: the memory limit is fixed at client start (env var), and it does not cover host fallback memory.
4. Without MPS (time slicing), memory is still shared spatially and faults stay contained. A no-MPS mode is the safer default for untrusted tenants.
5. Nixie (`eebf583`) contains no MPS or green-context code. **[src]**

## Open

- Can a running MPS client's memory limit change? Docs mention dynamic adjustment in "MPS v3"; not checked. Static SM partitioning needs CUDA 13.1 (we have 13.0).
- Does Nixie's sidecar run under MPS? Not tested.
- 1 to 3 runs each; repeat at least 5 times (median, range) before citing.
