# Nixie

[code](https://github.com/XOR-op/Nixie) | [arxiv](https://arxiv.org/html/2601.11743v1) | [presentation](https://www.youtube.com/watch?v=s-iTT7L9Y4o)

- focused on consumer workloads: hetero, rapidly changing, user-driven, batchy vs IOy
- each ML application saturates VRAM so need smarter GPU multiplexing

- idea:
    - keep one app (A) active at a time, and log all other apps' kernel launches as requests
    - ensure A's working set is VRAM resident
    - when B wants to be scheduled (from among the requests), talk to A's sidecar asking it to stop launching new kernels
    - prepare "plan": which of B's chunks need to be pulled into VRAM, which of A's need to be pulled out
    - wait for A to cuCtxSynchronize() on all threads
    - then do the memory operations
    - then finally B's kernel launch takes place
    - plus: "scheduler" detects idleness by tracking time since last CUDA API call; used to determine which request is processed first and whose blocks to kick out from lower tiers of memory backings

- benefits:
    - LD_PRELOAD (sidecar) based implementation - no application or driver changes required
    - two way memcpy's to better utilize throughput
    - feedback-based priority used for scheduling + residency

- lacks:
    - only handles cudaMalloc type calls; VMM calls are too low level to be handled by shim (ignored i guess)
    - importantly though, does not handle memadvise, memprefetch calls
    - 

## UVM notes
- Article: [How to maximize UVM performance](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/?_gl=1*2otcst*_gcl_au*OTc4MjYyOTQ2LjE3ODY4NzcwMTcuLS4tLjE3ODY4NzcwMTcuNTY1NjIzNzMuMTc5MDE2Mjk2OC4xNzkwMTYyOTY4)
- Survey paper: [link](https://tallendev.github.io/assets/papers/sc21.pdf) -- havent read it

### Nsight profiling
Use something like:
```
nsys profile \
  --trace=cuda,osrt,nvtx \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  -o uvm \
  ./app
```
Nsight Systems GUI, Open report:
- Bottom tab, Event View -> see summary of UVM page faults and memory transfers
- Main tab, Open CUDA HW > Unified Memory -> see HtoD and DtoH migrations (with cause - pf or pgfault)

### GPU side fault handling
From the article:
> The sequence of operations (assuming no cudaMemAdvise hints are set and there is no thrashing) is:
>
> 1. Allocate new pages on the GPU;
> 2. Unmap old pages on the CPU;
> 3. Copy data from the CPU to the GPU;
> 4. Map new pages on the GPU;
> 5. Free old CPU pages.
>
> Faults are cached in a buffer. Interrupt-based handling in CPU-side UVM driver code - allocates new pages, changes mappings, copies data

### Throughput underutilization
Regular UVM does one thing at a time - offload A or onload B - we can do both and get higher throughput

### Prefetching optimization
Article mentions some cudaMemPrefetchAsync workings that can be used to speed up independent kernels where D<->H data transfer is needed.
By ordering the kernel launch, prefetch onto GPU and pf onto CPU in the right way maximum "overlapping" can be achieved