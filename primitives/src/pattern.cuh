// fill / check a buffer with a known pattern: word i holds k * (i + 1), k != 0
// shared by the runner and the tests; both block until the kernel is done
#pragma once

#include <cuda_runtime.h>

namespace pattern {

__device__ unsigned long long d_bad;

__global__ void fill_k(unsigned *p, size_t n, unsigned k) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = k * (unsigned)(i + 1);
}
// ponytail: one atomic per wrong word, slow if a whole large buffer is wrong; per-block counts if that matters
__global__ void check_k(const unsigned *p, size_t n, unsigned k) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != k * (unsigned)(i + 1)) atomicAdd(&d_bad, 1ull);
}

// n words; returns the first launch or kernel error
inline cudaError_t fill(unsigned *p, size_t n, unsigned k) {
    fill_k<<<160, 256>>>(p, n, k);
    cudaError_t e = cudaGetLastError();
    return e != cudaSuccess ? e : cudaDeviceSynchronize();
}

// n words; wrong-word count in bad
inline cudaError_t check(const unsigned *p, size_t n, unsigned k, unsigned long long &bad) {
    bad = 0;
    cudaError_t e = cudaMemcpyToSymbol(d_bad, &bad, sizeof bad);
    if (e != cudaSuccess) return e;
    check_k<<<160, 256>>>(p, n, k);
    if ((e = cudaGetLastError()) != cudaSuccess || (e = cudaDeviceSynchronize()) != cudaSuccess) return e;
    return cudaMemcpyFromSymbol(&bad, d_bad, sizeof bad);
}

}  // namespace pattern
