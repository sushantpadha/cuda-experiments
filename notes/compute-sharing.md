# Compute sharing (MPS, green contexts)

> AI-generated :) probes in `experiments/lookups/2A/`

- no MPS: processes take turns on the GPU (time slicing), even with SMs free
- MPS: kernels from different processes really run at the same time
- green contexts: split SMs inside one process; across processes only useful under MPS (then they isolate SMs)
- VMM works under MPS: device + host memory, remap, handle export/import
- MPS caps (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`, `ACTIVE_THREAD_PERCENTAGE`) work on VMM device memory, not on host memory
- under MPS one client's bad access kills every client, busy or idle -> evicted memory must never be left unmapped
- remap while a kernel reads that buffer = crash; no atomic swap. unrelated kernels can keep running
- host-backed reads ~13 GB/s vs ~187 GB/s device
- CUDA 13 binaries launch via `__cudaLaunchKernel`, not `cudaLaunchKernel` (Nixie doesn't hook it)
- MPS pipe dir path must be short (unix socket limit), else the daemon dies silently
