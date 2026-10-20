// shared by the tests: fill/check pattern kernels, check printing, init/shutdown around a test body
#pragma once

#include "libvmem.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// fill / check a buffer with a known pattern: word i holds k * (i + 1), k != 0
// both block until the kernel is done
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

namespace t {

inline int fails = 0;

inline void expect(bool ok, const char *what) {
    printf("  %s: %s\n", ok ? "ok" : "FAIL", what);
    fails += !ok;
}

// f must throw vmem::Error with this status
template <class F> void expect_throws(vmem::Status s, F f, const char *what) {
    bool ok = false;
    try { f(); } catch (const vmem::Error &e) { ok = e.status == s; }
    expect(ok, what);
}

inline void put(void *p, size_t bytes, unsigned k) { pattern::fill((unsigned *)p, bytes / 4, k); }

inline bool intact(void *p, size_t bytes, unsigned k) {
    unsigned long long bad = 1;
    return pattern::check((unsigned *)p, bytes / 4, k, bad) == cudaSuccess && bad == 0;
}

// the backend and the daemon socket come from the test, not from VMEM_BACKEND or VMEM_DAEMON
// daemon == nullptr: no broker is used, the link must be Local
template <class F> int run(const char *name, vmem::Kind kind, F body, const char *daemon = nullptr) {
    unsetenv("VMEM_BACKEND");
    unsetenv("VMEM_DAEMON");
    try {
        vmem::init({.backend = kind, .daemon = daemon, .name = name});
        printf("backend: %s\n", vmem::get().info().backend);
        expect(vmem::get().info().link == (daemon ? vmem::Link::Connected : vmem::Link::Local),
               daemon ? "link to vmemd is connected" : "no vmemd: link is local");
        body();
        // a broker lost mid-test turns the link Broken and the backend goes on locally, so every check above could
        // pass without the daemon. this makes sure the whole body really ran against it
        if (daemon) expect(vmem::get().info().link == vmem::Link::Connected, "link to vmemd still connected at the end");
        vmem::shutdown();
    } catch (const vmem::Error &e) {
        printf("FAIL: uncaught %s (%s)\n", e.what(), vmem::status_name(e.status));
        return 2;
    }
    printf(fails ? "FAIL: %d check(s)\n" : "PASS\n", fails);
    return fails != 0;
}

}  // namespace t
