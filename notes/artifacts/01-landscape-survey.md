# Phase 1: landscape survey

Status: first pass, 2026-09-24. Owner: S. Tracker items: 2Aiii (survey), 2Ai (anchor cases).

Entry = stated purpose (abstract/intro), features, shortfall for `vmadvise`. "Lacks" = our analysis, not authors' claims. Tags: **[read]** read in full, **[abstract]** abstract/search result only, **[BoxD-cited]** known only from BoxD bibliography. Nothing citable in report until upgraded and added to `report/references.bib`. Links: `RESOURCES.md`.

Anchor cases (`TRACKER.md` 2Ai): **A** batch streaming, naive kernels; **C** multi-process residency; **D** grow-in-place buffers. Case F (cross-process sharing) parked.

## 1. BoxD [read]

- **Venue:** anonymous submission. Filename suggests SoCC winter 2026. Unconfirmed.
- **Purpose (abstract):** UVM eviction device-global, process-agnostic: oversubscribed co-located processes get non-deterministic residency, interference. BoxD puts managed memory in Linux cgroups for deterministic multi-tenant provisioning.
- **Mechanism:** modified `nvidia-uvm` module plus cgroup controller. Per-cgroup, per-GPU soft/hard/critical limits, eviction count. Lazy eviction: only on allocation failure. Victim in cgroup: heaviest process. Across cgroups: furthest over soft limit. Below soft limit: protected from other cgroups. Ships `nuvmtop` (per-process fault, eviction counters).
- **Evidence:** identical 16 GB tenants, 32 GB Blackwell: 7.5 to 12.6 million faults, 300 to 475 GB evicted per process (Figure 1). Victim mean runtime 11.5 s isolated, 233 s oversubscribed (Table 2), 16.0 s with BoxD. Victim faults stay near 1,000 whatever bully does (Table 3). Gold/Silver/Bronze tiers finish 297/81/78 requests. Pre-allocating victim soft limit before interference: 4-process run about 6 hours to 72 s (Section 5.2.3).
- **Lacks:**
  - Needs modified kernel driver, not a library.
  - Process/cgroup level. No per-buffer knowledge, no access-pattern hints. Paper lists application hints (prefetch, access pattern, offload) as future work (Section 7).
  - Reactive: no fault avoided, only redirected to right victim.
  - Evaluation synthetic plus cuGraph.
  - Still UVM: fault-driven migration.
- **Relevance:** framing for case C. Its gap = `vmadvise` position: userspace, VMM, application-level information.

## 2. Nixie [read]

Xu, Wang, Ren, Chen, Zhuo (Duke). arXiv:2601.11743v1, cs.OS, 16 Jan 2026. Preprint, review status unknown. Read in full from arXiv PDF (local, see `RESOURCES.md`).

Closest related work: userspace, CUDA VMM, no page faults.

- **Purpose (abstract):** consumer GPUs: single-user, fast-changing workloads, working sets nearly fill VRAM. UVM thrashes, overuses CPU pinned memory. Nixie multiplexes applications over time as system service, "without requiring any application or driver changes", MLFQ-inspired scheduler for interactive jobs.
- **Architecture:** `LD_PRELOAD` shim plus centralized daemon (about 10,000 lines Rust). Shim intercepts alloc/free, kernel and graph launches, `cudaMemGetInfo`, implicit allocators such as `cudaStreamCreate`. Kernels run in application's own CUDA context.
- **Memory model:**
  - Chunk = one application allocation, capped 128 MB (larger split). Built from 2 MB blocks, "the smallest physical allocation unit supported by the VMM API", up to 64 per chunk. Block = migration unit.
  - VMM reserves GPU virtual addresses, remaps to different physical allocations across migration; pointers stay valid.
  - Four tiers: GPU, CPU pinned, CPU paged, disk. One copy per chunk (not caches). GPU-resident chunks need no pinned backing, unlike UVM.
  - Migration: daemon global plan, shim async 2 MB block copies, evict and fetch overlap in both PCIe directions, small streaming window reserved for evicted application.
  - Safety: before migrating running application, block new launches, synchronize to drain kernels. Avoids invalid accesses. Coarse but correct.
- **Scheduling:** temporal only. One application's working set resident at a time. Idle = 100 ms timeout on API returns. MLFQ with soft recovery (T = 8 s demotion window, S = 4 s preemption, doubling per level). Next application's data prefetched during other's run.
- **Evidence (RTX 5090 32 GB, PCIe 5.0 x8, 96 GB DDR5, CUDA 12.9, driver 580.95.05):**
  - Context-switch time 29.1 to 82.3% lower than Ollama, UVM, nvshare. Time to first token 44.0 to 82.3% lower (Ollama cases), 29.7 to 36.3% lower (SGLang cases) than UVM and nvshare.
  - Bi-directional copy throughput about 2x UVM, near `nvbandwidth` max.
  - UVM-equal performance with 33.2 to 40.2% of CPU pinned memory (up to 66.8% saved).
  - 1.3 to 1.6x multi-application workflows, 3.1 to 3.8x interactive code completion vs. nvshare. Second machine (RTX A5000, PCIe 4.0 x16): 3.4x vs. nvshare.
  - No overhead on small ResNet inference. Allocations under 2 MB go to `cudaMalloc`; 2 to 128 MB slightly slower.
- **Case against UVM (Section 2.2):**
  1. No joint compute/memory control: thrashing (two 24 GB models on 32 GB, at least 16 GB migrated per forward pass, 250 ms at 64 GB/s vs. 20 to 75 ms pass).
  2. Half-duplex use of full-duplex link: fault evicts first, then fetches.
  3. LRU metadata updates only on faults, "a highly incomplete view" of accesses (cites Allen and Ge, IPDPS 2021).
  4. Every GPU-resident page needs pinned CPU page.
  Per-fault interrupt latency not quantified: argument is thrashing, half-duplex transfer, pinned-memory cost, not fault stalls.
- **Baseline caveat:** "UVM" baseline hooks only `cudaMalloc`, `cudaFree`, `cudaMemGetInfo`. No `cudaMemAdvise`, no `cudaMemPrefetchAsync`. No win shown over tuned UVM.
- **Stated limits, future work:** white-box hints "an interesting direction for future work" (Section 8). Spatial multiplexing of small models future work. Single-user threat model, no isolation. UVM oversubscription unsupported on Windows, motivating VMM.
- **Lacks (our analysis):**
  - Transparent, so no application semantics: no per-buffer hints, priorities, access schedules. Whole chunks move between whole-application slots.
  - Temporal multiplexing assumes one working set fits VRAM. Ignores single application exceeding VRAM (case A), growing buffers (case D).
  - No quotas, tenant isolation.
- **Consequence:** Nixie covers temporal case C. A, D untouched. Claim for C must be positioned against BoxD (kernel, UVM) and Nixie (userspace, VMM, transparent, temporal). Settle in Phase 1.

## 3. GMLake [abstract]

Guo, Zhang, Xu, Leng, Liu, Huang, Guo, Wu, Zhao, Zhao, Zhang. ASPLOS 2024. arXiv:2401.08156.

- **Purpose:** PyTorch/TensorFlow caching allocators "degrade quickly" with recomputation, offloading, distributed training; fragmentation follows. GMLake fuses non-contiguous physical blocks into one virtual range ("virtual memory stitching") via CUDA low-level VMM API, transparent to models.
- **Claimed:** 9.2 GB average (up to 25 GB) less GPU memory, 15% (up to 33%) less fragmentation, eight LLMs, A100 80 GB. Search summary also mentioned about 10x overhead for native allocation path. Check abstract before using.
- **Lacks:** single device, training-oriented, pure fragmentation fix. No host tier, residency hints, oversubscription handling, multi-tenant control.
- **Relevance:** VMM stitching works in production. Related to case D, PyTorch study (`experiments/pytorch-vmm-study/`).

## 4. vAttention [abstract]

Prabhu, Nayak, Mohan, Ramjee, Panwar. ASPLOS 2025 per arXiv listing. arXiv:2405.04437.

- **Purpose:** PagedAttention makes KV cache non-contiguous in virtual memory, needs custom kernels. vAttention keeps KV cache virtually contiguous, maps physical memory on demand via CUDA VMM; unmodified attention kernels work.
- **Claimed:** up to 1.23x throughput over PagedAttention-based FlashAttention and FlashInfer kernels. Abstract mentions "LLM-specific optimizations to address the limitations of CUDA virtual memory support", unlisted. Read paper for list: VMM limits `vmadvise` will hit too.
- **Lacks:** KV cache only. Grow/free, no eviction, host tier, multi-tenant policy.
- **Relevance:** direct precedent for case D (grow in place, page-level physical backing).

## 5. vTensor [abstract]

arXiv:2407.15309. Search snippets only.

- **Purpose:** VMM-based tensor structure for LLM inference, decouples computation from memory defragmentation, CPU-side management of physical handles.
- **Claimed (snippet):** 1.86x average speedup, up to 2.42x multi-turn chat, about 71% (57 GB) memory freed on A100 vs. vLLM.
- **Lacks:** like vAttention: LLM-specific, no tiering or multi-tenant control.

## 6. RAPIDS RMM [docs]

- **Purpose:** composable device memory resources for RAPIDS.
- **Features:** managed-memory resource, pool and arena resources, adaptors for prefetch (`prefetch_resource_adaptor` prefetches every allocation), limiting, logging, tracking. BoxD's cuGraph server used RMM prefetch adaptor over managed pool.
- **Lacks:** prefetch per allocation, not scheduled or hinted per range. Limits = byte caps, not residency control. UVM or `cudaMalloc`, not VMM residency.
- **Relevance:** baseline, possible integration point. Only docs pages read.

## 7. Other related work

- **BoxD-cited [BoxD-cited]:** Allen et al. (TACO 2024, UVM demand-paging analysis), Ganguly et al. (IPDPS 2020, adaptive migration under oversubscription), DeepUM (ASPLOS 2023), SUV (MICRO 2024), Forest (ISCA 2025), Early-adaptor (ISPASS 2023), DREAM (ICS 2025), Choi et al. (ATC 2022), MuxFlow, SGDRC, Fractional GPUs. All UVM-mechanism or compute-sharing.
- **Nixie-cited [Nixie-cited]:** nvshare (UVM time slicing), TGS (priority sharing on UVM), Prism, Aegaeon, ServerlessLLM (datacenter model swapping), G10 (UVM plus storage), XSched (preemptive scheduling), Allen and Ge (IPDPS 2021, UVM cost analysis).
- **Driver-side policy [abstract]:** gpu_ext (arXiv 2512.12615) adds eBPF hooks for eviction and prefetch policy to nvidia-uvm. Same goal as vmadvise hints, opposite layer. See `uvm-driver/README.md`.
- **Handoff list, unverified:** vDNN, Capuchin, SwapAdvisor, Salus, AntMan, Zico, ZeRO-Infinity, FlexGen, GPUswap, Gdev, PTask, Waldspurger (ESX), HMM and ATS behaviour, MPS and MIG details. Trim to what we cite before verifying.

## 8. Comparison table

| Work | Layer | Memory API | Multi-tenant | App-level hints | Host tier | Avoids fault stalls |
|---|---|---|---|---|---|---|
| UVM (baseline) | Driver | Managed | No (device-global) | `cudaMemAdvise`, prefetch | Yes | No |
| BoxD | Kernel driver | Managed | Yes (cgroup) | Future work | Yes (UVM) | No |
| Nixie | Userspace service | VMM plus pinned copies | Temporal only | Future work | Yes (pinned, paged, disk) | Yes |
| GMLake | Userspace allocator | VMM | No | No | No | n/a |
| vAttention, vTensor | Userspace, LLM | VMM | No | No | No | n/a |
| RMM | Userspace library | Managed or `cudaMalloc` | No | Prefetch adaptor | Via managed | No |
| `vmadvise` (goal) | Userspace library | VMM | Case C, later | Yes | Yes | Yes (see `remap-vs-uvm.md`) |

## 9. Claimed gap (draft, to be corrected)

BoxD, Nixie show residency control matters under oversubscription. BoxD: kernel, UVM. Nixie: transparent, userspace, VMM. Neither takes application-supplied information (which buffers matter, use order, growth). `vmadvise` claims niche: explicit userspace interface carrying this into VMM-based residency control.

Risk: do hints beat Nixie-style transparency enough to justify interface. Known downside: no access-frequency information (see `notes/limitations.md`).
