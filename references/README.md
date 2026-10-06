# References

Documentation and literature used by the project. PDFs are kept locally in this folder and are not tracked by git.

## Documentation

- **Key reference.** NVIDIA blog, Introducing Low-Level GPU Virtual Memory Management (Perry and Sakharnykh, 2020): <https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/>. Old, but the mechanism still holds; summary and sync behaviour in `notes/vmm-api.md`.
- CUDA Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>; sections on [Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html), [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html), [Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- CUDA Driver API, Virtual Memory Management: <https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html>
- CUDA Runtime API, memory management: <https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html>; synchronization behaviour: <https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html>
- PyTorch CUDA caching allocator source: <https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp>
- PyTorch memory snapshot tooling: <https://docs.pytorch.org/docs/stable/torch_cuda_memory.html>
- Linux `madvise(2)`: <https://man7.org/linux/man-pages/man2/madvise.2.html>

## Related systems

- GMLake, ASPLOS 2024: <https://arxiv.org/pdf/2401.08156>
- vTensor: <https://arxiv.org/pdf/2407.15309>
- antgroup/glake: <https://github.com/antgroup/glake>

## Local only

- `boxd_socc_winter2026_v3 1.pdf`: an anonymous manuscript on managing GPU managed memory. Not redistributed.
- `iitm-gpu-course-slides/`: lecture slides from a GPU course. Not redistributed.
- `cuda-programming-guide.pdf`: local copy of the guide above.
