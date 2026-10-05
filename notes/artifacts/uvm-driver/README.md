# UVM driver: how it works, and where to read about it

Started 2026-09-27. Question that opened it: does UVM keep host memory behind pages that live on the GPU (Nixie §2.2 claims yes)? Related tracker items: 5D (read UVM driver source), 5B/5C (UVM vs. VMM experiments).

Tags: **[src]** read in driver source this session; **[paper]** stated by the cited paper (read); **[abstract]** only abstract or search result seen; **[untested]** not run on our machine.

## 1. Answer to the opening question

Short: **yes, in the common case, but not as a live second copy.**

- UVM migration is move, not duplicate. After a CPU→GPU migration the CPU copy is no longer "resident" (not coherent). Only read-duplication (`cudaMemAdviseSetReadMostly`) keeps valid copies on several processors. **[src]** comment on `resident` in `uvm_va_block.h:234-247`.
- But the CPU *page* is not freed. The same comment: a cleared resident bit means "A CPU chunk for the corresponding page index may or may not have been allocated. If the chunk is present, it's a cached chunk which can be reused in the future." Allocation is tracked in a separate mask, `allocated` (`uvm_va_block.h:249-253`). **[src]**
- CPU chunks of managed (non-HMM) memory are freed only when the VA block is destroyed (`cudaFree`, `uvm_va_block.c:~9730`), when a block is split, or on an allocation error path. The HMM path has its own `uvm_va_block_remove_cpu_chunks`. No path frees them on migration to the GPU, and the module registers no memory shrinker. **[src]** (grep of `uvm_cpu_chunk_free`, `shrinker` in `kernel-open/nvidia-uvm/`).
- The pages come from `alloc_pages()` in the driver (`uvm_pmm_sysmem.c:444-472`), so they are kernel-owned pages, not user anonymous memory: the OS cannot swap them. That is what Nixie calls "pinned". **[src]**
- A page first touched on the GPU gets no CPU chunk until it is evicted to the CPU. So the host cost applies to data that has ever been on the CPU (for example model weights loaded by the CPU) or has been evicted once. **[src]** (inferred from the allocation paths; **[untested]**).

So Nixie's "every page resident on the GPU must have a corresponding pinned page on the CPU" overstates it (first-touch-on-GPU pages have none), but for their workloads, which load weights on the CPU first, the host pages stay reserved while the data sits on the GPU. This matches their Fig. 9.

Test to confirm on our machine (user-driven, not run): `cudaMallocManaged` N GB, write it on the CPU, `cudaMemPrefetchAsync` to the GPU, sync, then compare `MemAvailable` in `/proc/meminfo` before and after. Expected: host memory not returned until `cudaFree`.

## 2. How the driver works (short map)

Source: `kernel-open/nvidia-uvm/` in NVIDIA open-gpu-kernel-modules, tag `580.178.04` (the version installed here). Sparse clone lived in the session scratchpad; re-clone with `git clone --depth 1 --branch 580.178.04 --filter=blob:none --sparse https://github.com/NVIDIA/open-gpu-kernel-modules && git sparse-checkout set kernel-open/nvidia-uvm`.

| Topic | Fact | Where |
|---|---|---|
| VA block | Managed allocations are split into 2 MB VA blocks; each tracks per-processor residency, mappings, CPU chunks, GPU chunks | `uvm_va_block.h`, `uvm_va_block_types.h`; SC21 §2.2 **[paper]** |
| Fault path | GPU writes faults to a hardware buffer; driver fetches them in batches of up to 256 (`uvm_perf_fault_batch_count`), groups by VA block, services each block, then replays | `uvm_gpu_replayable_faults.c:73`; SC21 §2.2 **[src][paper]** |
| Prefetch | Tree-based density prefetcher inside a VA block; threshold 51% (`uvm_perf_prefetch_threshold`) | `uvm_perf_prefetch.c:42` **[src]** |
| Thrashing | Detection plus throttling or pinning of pages that bounce | `uvm_perf_thrashing.c` (not yet read) |
| GPU eviction | GPU memory in 2 MB root chunks on an LRU list `va_block_used`; a root chunk moves to the tail when a VA block *allocates* in it, not on access. So "LRU" is by migration/allocation time | `uvm_pmm_gpu.c:99-113` **[src]** |
| Access counters | Hardware access counters can drive migration (`uvm_perf_access_counter_*` params) | `uvm_gpu_access_counters.c` (not yet read; tracker 5D) |
| CPU pages | `alloc_pages`, 4K/64K/2M chunks, cached after migration (§1) | `uvm_pmm_sysmem.c`, `uvm_va_block.h` **[src]** |
| Tunables | `/sys/module/nvidia_uvm/parameters/*` on this machine: prefetch threshold 51, fault batch 256 (read 2026-09-27) | **[src]** |

## 3. Sources, in reading order

Driver-internals first, then measurement, then policy research.

1. **NVIDIA driver source**, `kernel-open/nvidia-uvm/` (tag 580.178.04). Primary truth. Start with `uvm_va_block.h` comments, then `uvm_gpu_replayable_faults.c`, `uvm_pmm_gpu.c` header comment, `uvm_perf_prefetch.c`. <https://github.com/NVIDIA/open-gpu-kernel-modules>
2. **Allen, Ge. In-Depth Analyses of Unified Virtual Memory System for GPU Accelerated Computing.** SC21. doi 10.1145/3458817.3480855. PDF: <https://tallendev.github.io/assets/papers/sc21.pdf>. Fault batches, 2 MB VABlocks, eviction at VABlock granularity, instrumented driver 460.27.04 on a Titan V. **[paper]** (sections 2-3 read).
3. **Allen, Ge. Demystifying GPU UVM Cost with Deep Runtime and Workload Analysis.** IPDPS 2021, pp. 141-150, doi 10.1109/IPDPS49936.2021.00023. Where fault-handling time goes; prefetcher behaviour. Cited by Nixie as [4]. **[abstract]**
4. **Allen, Cooper, Ge. Fine-grain Quantitative Analysis of Demand Paging in Unified Virtual Memory.** ACM TACO 21(1), doi 10.1145/3632953. Open copy: <https://par.nsf.gov/biblio/10566579>. Cost of each step on the fault command and data paths. **[abstract]**
5. **tallendev/uvm-eval** (reproducibility repo for 2 and 3, instrumented driver): <https://github.com/tallendev/uvm-eval>. **njones93531/uvm-eviction** (instrumented driver logging faults, prefetches, evictions): <https://github.com/njones93531/uvm-eviction>. **[abstract]**
6. **NVIDIA blogs (Sakharnykh).** *Maximizing Unified Memory Performance in CUDA*: <https://developer.nvidia.com/blog/maximizing-unified-memory-performance-in-cuda/>. *Improving GPU Memory Oversubscription Performance*: <https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/>. *Beyond GPU Memory Limits with Unified Memory on Pascal*: <https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/>. Official, user-level view of migration, prefetch, advise. **[abstract]**
7. **CUDA Programming Guide, Unified Memory chapter.** Already in `references/`. Semantics of advise, prefetch, read-duplication.
8. **Zheng, Nellans, Zulfiqar, Stephenson, Keckler. Towards High Performance Paged Memory for GPUs.** HPCA 2016, pp. 345-357. PDF: <https://www.cs.utexas.edu/~skeckler/pubs/HPCA_2016_Paged_Memory.pdf>. Foundational: why demand paging is slow over PCIe and how to hide it. **[abstract]**
9. **Ganguly, Zhang, Yang, Melhem. Interplay between Hardware Prefetcher and Page Eviction Policy in CPU-GPU Unified Virtual Memory.** ISCA 2019, doi 10.1145/3307650.3322224. Prefetching turns harmful under oversubscription with locality-unaware eviction; tree-based pre-eviction. Simulator: <https://github.com/DebashisGanguly/gpgpu-sim_UVMSmart>. **[abstract]**
10. **Kim, Sim, Gera, et al. Batch-Aware Unified Memory Management in GPUs for Irregular Workloads.** ASPLOS 2020, doi 10.1145/3373376.3378529. Batched fault handling serialises execution. **[abstract]**
11. **Shao, Guo, Wang, Wang, Li, Guo. Oversubscribing GPU Unified Virtual Memory: Implications and Suggestions.** ICPE 2022 (best paper), doi 10.1145/3489525.3511691. PDF: <https://cs.sjtu.edu.cn/~lichao/publications/Oversubscribing_GPU_ICPE-2022-Shao.pdf>. Why workloads differ in oversubscription sensitivity. **[abstract]**
12. **Zheng et al. gpu_ext: Extensible OS Policies for GPUs via eBPF.** arXiv 2512.12615. Adds eBPF hooks for eviction and prefetch policy to nvidia-uvm (open modules 575.57.08). <https://github.com/eunomia-bpf/gpu_ext>. **Directly relevant to libvmem**: a driver-side way to inject residency policy. **[abstract]** (repo README read).
13. **Nazaraliyev, Sadredini. GPUVM: GPU-driven Unified Virtual Memory.** arXiv 2411.05309. **[abstract]**
14. Informal: Pranjal Singh, *Studying memory migration in NVIDIA open source driver*, IIT Kanpur CS614 report: <https://cse.iitk.ac.in/users/prsingh/projects/cs614-final-report.pdf>. Student report; useful walkthrough, do not cite as authority. **[abstract]**

## 4. Open

- Read `uvm_perf_thrashing.c` and `uvm_gpu_access_counters.c`: do access counters change eviction order on this GPU? (tracker 5D)
- Run the host-memory test in §1.
- Does `uvm_global_oversubscription` or HMM mode change the CPU-chunk caching? Not checked.
- Read papers 3 and 4 in full: they have the fault-path cost breakdown asked for in the landscape survey.
