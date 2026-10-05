#include "vmem.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define TIME_START(id) auto t_start_##id = std::chrono::high_resolution_clock::now()
#define TIME_END(id) printf("\t\t%.3f ms\n", std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start_##id).count())

__global__ void fill(unsigned *p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (unsigned)i * 1729u + seed;
}

__global__ void check(const unsigned *p, size_t n, unsigned seed, unsigned long long *bad) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != (unsigned)i * 1729u + seed) atomicAdd(bad, 1ull);
}

int main(int argc, char **argv) {
    int a = argc > 1 ? atoi(argv[1]) : 12;
    int b = argc > 2 ? atoi(argv[2]) : 24;
    const size_t MiB = 1 << 20;
    const unsigned seed = 420u;
    const size_t bytes = sizeof(unsigned);

    printf("====================== TEST %dMiB %dMiB ======================\n", a, b);
    printf("stream 1: fill %d MiB on device then copy to host with remap\n", a);
    printf("stream 2: fill %d MiB on device then copy to host with memcpy\n", b);
    printf("==============================================================\n");

    try {
        vmem::Manager m(vmem::Options{});

        TIME_START(001);
        auto vad = m.reserve(a * MiB);
        auto vbd = m.reserve(b * MiB);
        auto vbh = m.reserve(b * MiB);
        TIME_END(001);

        TIME_START(002);
        auto ad = m.create(a * MiB, vmem::Loc::Device);
        auto bd = m.create(b * MiB, vmem::Loc::Device);
        auto bh = m.create(b * MiB, vmem::Loc::Host);
        TIME_END(002);

        TIME_START(003);
        m.map(ad, vad);
        m.map(bd, vbd);
        m.map(bh, vbh);
        TIME_END(003);

        CUstream s1, s2;
        TIME_START(004);
        cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING);
        cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING);
        TIME_END(004);

        TIME_START(005);
        fill<<<160, 256, 0, s1>>>(m.ptr<unsigned>(ad), a * MiB / bytes, seed);
        m.remap(ad, vmem::Loc::Host);
        cudaStreamSynchronize(s1);
        TIME_END(005);

        TIME_START(006);
        fill<<<160, 256, 0, s2>>>(m.ptr<unsigned>(bd), b * MiB / bytes, seed);
        cudaMemcpyAsync(m.ptr<unsigned>(bh), m.ptr<unsigned>(bd), b * MiB, cudaMemcpyDeviceToHost, s2);
        cudaStreamSynchronize(s2);
        TIME_END(006);

        unsigned long long bad = 0;

        TIME_START(007);
        check<<<160, 256, 0, s1>>>(m.ptr<unsigned>(ad), a * MiB / bytes, seed, &bad);
        check<<<160, 256, 0, s2>>>(m.ptr<unsigned>(bh), b * MiB / bytes, seed, &bad);
        cudaStreamSynchronize(s1);
        cudaStreamSynchronize(s2);
        TIME_END(007);

        if (bad != 0) {
            fprintf(stderr, "FAIL: %llu bad words\n", bad);
            return 1;
        } else {
            fprintf(stderr, "PASS\n");
            return 0;
        }

    } catch (const vmem::Error &e) {
        fprintf(stderr, "UNEXPECTED vmem::Error: %s\n", e.what());
        return 2;
    }
}