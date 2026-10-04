# Resources

Sources. Update on read or verify.
Status: **read** (full text), **abstract** (abstract or search result only), **cited** (only via another paper's bibliography), **docs** (product docs).
Not citable in `report/` until read, verified, added to `report/references.bib`.

## Papers and preprints

| Source | Status | Link or location |
|---|---|---|
| BoxD: Managing GPU Managed Memory (anonymous, venue unconfirmed) | read | `references/boxd_socc_winter2026_v3 1.pdf` (local only) |
| Nixie: Efficient, Transparent Temporal Multiplexing for Consumer GPUs (Xu et al.; arXiv 2601.11743) | read | https://arxiv.org/abs/2601.11743 ; code https://github.com/XOR-op/Nixie (read at eebf583, see `nixie-summary/`) |
| Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning (Yu et al.; OSDI '26; arXiv 2505.04021) | read | `references/prism-paper.pdf` (local only); code https://github.com/ovg-project/kvcached |
| GMLake (Guo et al.; ASPLOS 2024; arXiv 2401.08156) | abstract | https://arxiv.org/abs/2401.08156 , ACM https://dl.acm.org/doi/10.1145/3620665.3640423 |
| vAttention (Prabhu et al.; arXiv 2405.04437) | abstract | https://arxiv.org/abs/2405.04437 |
| vTensor (arXiv 2407.15309) | abstract (snippets) | https://arxiv.org/abs/2407.15309 |
| Allen and Ge, Demystifying GPU UVM cost (IPDPS 2021) | abstract | doi 10.1109/IPDPS49936.2021.00023. See `uvm-driver/README.md`. |
| Allen, Cooper, Ge, Fine-grain analysis of demand paging in UVM (TACO 2024) | cited by BoxD | doi 10.1145/3632953 (BoxD bib.) |
| Ganguly et al., Adaptive page migration under GPU oversubscription (IPDPS 2020) | cited by BoxD | doi 10.1109/IPDPS47924.2020.00054 |
| DeepUM (ASPLOS 2023), SUV (MICRO 2024), Forest (ISCA 2025), Early-adaptor (ISPASS 2023), DREAM (ICS 2025), Choi et al. (USENIX ATC 2022) | cited by BoxD | BoxD bib. |
| nvshare (Alexopoulos and Mitropoulos, ICSE 2024; repo https://github.com/grgalex/nvshare), TGS, Aegaeon, ServerlessLLM, G10, XSched | cited by Nixie | Nixie bib. |
| ObservUVM (ISCA 2026), SUV, OASIS (HPCA 2025), ARIADNE (HPCA 2026), MSched (2026) | secondary page only | via https://eunomia.dev/research/gpu-memory-placement-evidence/ . Verify at source. |
| Handoff list: vDNN, Capuchin, SwapAdvisor, Salus, AntMan, Zico, ZeRO-Infinity, FlexGen, GPUswap, Gdev, PTask, Waldspurger (ESX), Zheng et al. (HPCA 2016), Kim et al. (ASPLOS 2020) | not checked | from memory. Trim, verify. |

## Documentation and code

| Source | Status | Link or location |
|---|---|---|
| RMM memory resource adaptors | docs | https://docs.nvidia.com/rmm/26.08/cpp/memory_resources/memory_resource_adaptors/index.html |
| RMM repository | docs | https://github.com/rapidsai/rmm |
| RMM prefetch adaptor source | docs | https://docs.rapids.ai/api/librmm/26.02/prefetch__resource__adaptor_8hpp_source |
| antgroup/glake (GMLake code) | not read | https://github.com/antgroup/glake |
| microsoft/vattention (vAttention code) | not read | https://github.com/microsoft/vattention |
| NVIDIA open-gpu-kernel-modules, `uvm_pmm_gpu.c` (PMM, eviction) | read in part (header comment, eviction list) | https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia-uvm/uvm_pmm_gpu.c ; tag 580.178.04 matches this machine |
| NVIDIA open-gpu-kernel-modules, `uvm_gpu_access_counters.c` | to read | https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia-uvm/uvm_gpu_access_counters.c |
| CUDA Programming Guide (UVM, VMM, prefetch and advise) | read in part | `references/cuda-programming-guide.pdf` (local only) |
| Eunomia and DeepWiki write-ups of UVM driver | secondary, unverified | https://eunomia.dev/zh/blog/posts/nvidia-open-driver-analysis/ , https://deepwiki.com/NVIDIA/open-gpu-kernel-modules/4.5-uvm-memory-management |
| UVM driver study: sources, reading order, findings | index | `uvm-driver/README.md` (SC21 Allen and Ge read in part; gpu_ext arXiv 2512.12615, eBPF policy hooks in nvidia-uvm) |
| Secondary claims, leads only (fault latency 16-18 us, ObservUVM 34% speedup) | unverified | https://eunomia.dev/research/gpu-memory-placement-evidence/ , search results for "GPU page fault latency" |

## Local project material

- `report/sections/03_uvm_vs_vmm.tex`: UVM vs. VMM section.
- `experiments/pytorch-vmm-study/`: allocator study (branch `pytorch-study`).
- `experiments/VMMVector`, `experiments/VMMRemapShared`: scratch VMM (source of preliminary numbers, "cannot map partially", "no fault handler").
- Earlier scratch (`VMMExplore`, `VMMSlab`): branch `vmm-experiments`.

## People

- **Soham:** page-fault research. Ask: per-fault cost, batching, replayable faults, prefetch granularity. See `remap-vs-uvm.md`.
- **Prof. Purushottam Kulkarni:** mentor. Ask for 7B (BoxD manuscript in `references/`: origin, citation) and BoxD positioning.
