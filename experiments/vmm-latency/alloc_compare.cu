// times cudaMalloc / cudaMallocManaged against VMM, each VMM phase timed on its own:
//   vmm:     reserve | create + map | set access | unmap + release | address free
//   managed: cudaMallocManaged | touch kernel (one write per 4 KiB page) + sync | cudaFree
//   malloc:  cudaMalloc | cudaFree
// set access is not one-time: every cuMemMap needs it again before a kernel can touch the range
// plain host timer (no nsys); prints csv, unused columns are 0
// usage: ./alloc_compare <run> [reps] [gap_ms]   (run seeds the shuffle)
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

#define CU(call) do { CUresult e_ = (call); if (e_ != CUDA_SUCCESS) { const char *n_; cuGetErrorName(e_, &n_); \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, n_); exit(1); } } while (0)
#define RT(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorName(e_)); exit(1); } } while (0)

using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

// one write per 4 KiB page, so every page gets backed
__global__ void touch(char *p, size_t pages) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < pages; i += (size_t)gridDim.x * blockDim.x)
        p[i << 12] = 1;
}

int main(int argc, char **argv) {
    const int run = argc > 1 ? atoi(argv[1]) : 0;
    const int reps = argc > 2 ? atoi(argv[2]) : 10, warmup = 3;
    const int gap_ms = argc > 3 ? atoi(argv[3]) : 20;   // busy-wait between reps, as in vmm_latency's spaced mode
    const size_t sizes_mib[] = {2, 8, 32, 128, 512, 1024, 2048, 4096};
    const char *apis[] = {"cudaMalloc", "cudaMallocManaged", "vmm"};

    RT(cudaFree(0));   // runtime makes the primary context current; driver calls below use it too
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    CUmemAccessDesc a = {};
    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // ---- every (api, size, rep) in shuffled order ----
    struct Run { int api, rep; size_t mib; };
    std::vector<Run> runs;
    for (int api = 0; api < 3; ++api)
        for (size_t mib : sizes_mib)
            for (int r = 0; r < reps; ++r) runs.push_back({api, r, mib});
    std::shuffle(runs.begin(), runs.end(), std::mt19937(run));
    for (int w = 0; w < warmup; ++w)
        for (int api = 0; api < 3; ++api) runs.insert(runs.begin(), Run{api, -1, 2});   // rep -1 = warmup

    printf("run,api,size_mib,rep,reserve_us,alloc_us,set_access_us,touch_us,free_us,addr_free_us\n");
    for (const Run &x : runs) {
        const size_t sz = x.mib << 20;
        double t[6] = {};   // same order as the csv columns
        clk::time_point c[6];
        if (x.api == 0) {
            void *ptr;
            c[0] = clk::now(); RT(cudaMalloc(&ptr, sz));
            c[1] = clk::now(); RT(cudaFree(ptr));
            c[2] = clk::now();
            t[1] = us(c[0], c[1]); t[4] = us(c[1], c[2]);
        } else if (x.api == 1) {
            char *ptr;
            c[0] = clk::now(); RT(cudaMallocManaged(&ptr, sz));
            c[1] = clk::now(); touch<<<160, 256>>>(ptr, sz >> 12); RT(cudaDeviceSynchronize());
            c[2] = clk::now(); RT(cudaFree(ptr));
            c[3] = clk::now();
            t[1] = us(c[0], c[1]); t[3] = us(c[1], c[2]); t[4] = us(c[2], c[3]);
        } else {
            CUdeviceptr va; CUmemGenericAllocationHandle h;
            c[0] = clk::now(); CU(cuMemAddressReserve(&va, sz, 0, 0, 0));
            c[1] = clk::now(); CU(cuMemCreate(&h, sz, &p, 0)); CU(cuMemMap(va, sz, 0, h, 0));
            c[2] = clk::now(); CU(cuMemSetAccess(va, sz, &a, 1));
            c[3] = clk::now(); CU(cuMemUnmap(va, sz)); CU(cuMemRelease(h));
            c[4] = clk::now(); CU(cuMemAddressFree(va, sz));
            c[5] = clk::now();
            t[0] = us(c[0], c[1]); t[1] = us(c[1], c[2]); t[2] = us(c[2], c[3]);
            t[4] = us(c[3], c[4]); t[5] = us(c[4], c[5]);
        }
        if (x.rep >= 0)
            printf("%d,%s,%zu,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", run, apis[x.api], x.mib, x.rep,
                   t[0], t[1], t[2], t[3], t[4], t[5]);
        if (gap_ms) {
            auto e = clk::now() + std::chrono::milliseconds(gap_ms);
            while (clk::now() < e) {}
        }
    }
    return 0;
}
