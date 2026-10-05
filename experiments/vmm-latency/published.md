# Published VMM call latencies

Numbers others report, for comparison with `results/latency.csv`. Hardware differs from ours (RTX 4050 Laptop); treat as order-of-magnitude.

## vAttention (Prabhu et al., ASPLOS 2025), Table 3

Microseconds per call. `cu*` = stock CUDA (2 MB pages only); the 64 KB to 256 KB columns are the paper's own driver extension, not stock CUDA. GPU not stated next to the table.

| Call | 2 MB (stock CUDA) |
|---|---|
| cuMemAddressReserve | 2 |
| cuMemCreate | 29 |
| cuMemMap | 2 |
| cuMemSetAccess | 38 |
| cuMemUnmap | 34 |
| cuMemRelease | 23 |
| cuMemAddressFree | 1 |

Source: <https://arxiv.org/abs/2405.04437> (HTML v3, Table 3).

## GMLake (Guo et al., ASPLOS 2024), Table 1

Allocating 2 GB in chunks of a given size; each call's total time normalised to one `cuMalloc` of the same 2 GB (so 1.0 = as slow as `cuMalloc`). A100 testbed.

| Call | 2 MB chunks | 128 MB chunks | 1024 MB chunks |
|---|---|---|---|
| cuMemReserve | 0.003 | 0.003 | 0.002 |
| cuMemCreate | 18.1 | 0.89 | 0.79 |
| cuMemMap | 0.70 | 0.01 | 0.002 |
| cuMemSetAccess | 96.8 | 8.2 | 0.7 |
| Total | 115.4 | 9.1 | 1.5 |

Takeaway in the paper: 2 GB built from 2 MB chunks is 115x slower than one `cuMalloc`, and `cuMemSetAccess` dominates.

Source: <https://arxiv.org/abs/2401.08156> (Section 2.3, Table 1).

## NVIDIA blog, "Introducing Low-Level GPU Virtual Memory Management"

No timings. States that `cuMemSetAccess` costs `O(lg N)` in the number of allocations for the single-device case, and that VMM avoids the implicit synchronisation of `cudaFree`.

Source: <https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/>
