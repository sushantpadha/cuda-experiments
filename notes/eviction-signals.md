# Eviction signals

> AI-generated :) details in `notes/artifacts/lookups/2B-eviction-metrics/`

- no hardware access bits for VMM memory; access counters stuck inside nvidia-uvm, unusable
- launch-argument scan: which buffers a kernel may touch. inject via CUPTI (`CUDA_INJECTION64_PATH`), sees cuBLAS too; LD_PRELOAD misses library kernels. cost is low
- blind spot: pointers stored in GPU memory. host fallback makes a miss slow, not fatal
- kernel time grows steadily as more data is on host -> can be a per-tenant "pain" signal (noisy under MPS)
- promote if touched more than ~once while resident (one pass on host costs about one copy)
- block hashing on GPU ~188 GB/s, ~14x faster than copying out -> find unchanged blocks, drop them free
- profile-guided hints: run the app once under a Compute Sanitizer API (installed) or NVBit profiler, record which buffers each kernel really touches; the daemon then uses that profile as automatic hints. exact but ~10-100x slower, so one-off only; also checks the arg scan
