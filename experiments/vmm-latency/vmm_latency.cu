// times each VMM call for a range of sizes, device vs host, rw vs read-only
// run under nsys (see run.sh); one nvtx range per repetition labels the calls
// all reps of all configs run in one shuffled order (fixed seed)
#include <cuda.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>
#include <chrono>

#define CU(call) do { CUresult e_ = (call); if (e_ != CUDA_SUCCESS) { const char *n_; cuGetErrorName(e_, &n_); \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, n_); exit(1); } } while (0)

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20, warmup = 3;
    const int gap_ms = argc > 2 ? atoi(argv[2]) : 0;   // pause between reps (busy-wait: keeps the CPU awake)
    const size_t sizes_mib[] = {2, 8, 32, 128, 512, 1024};

    CUdevice dev; CUcontext ctx;
    CU(cuInit(0)); CU(cuDeviceGet(&dev, 0));
    CU(cuDevicePrimaryCtxRetain(&ctx, dev)); CU(cuCtxSetCurrent(ctx));
    int numa = 0; cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev); if (numa < 0) numa = 0;

    // ---- every (config, rep) in shuffled order, so slow driver states don't pile onto one config ----
    struct Run { int host, ro, rep; size_t mib; };
    std::vector<Run> runs;
    for (int host = 0; host < 2; ++host)
        for (size_t mib : sizes_mib)
            for (int ro = 0; ro < 2; ++ro)
                for (int r = 0; r < reps; ++r) runs.push_back({host, ro, r, mib});
    std::shuffle(runs.begin(), runs.end(), std::mt19937(42));
    for (int w = 0; w < warmup; ++w) runs.insert(runs.begin(), Run{0, 0, -1, 2});   // rep -1 = warmup

    for (const Run &x : runs) {
        const size_t sz = x.mib << 20;
        CUmemAllocationProp p = {};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type = x.host ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = x.host ? numa : 0;
        CUmemAccessDesc a = {};
        a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        a.location.id = 0;
        a.flags = x.ro ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        char label[64];
        snprintf(label, sizeof label, "%s|%zu|%s|%d", x.host ? "host" : "device", x.mib, x.ro ? "ro" : "rw", x.rep);
        nvtxRangePushA(label);
        CUdeviceptr va; CUmemGenericAllocationHandle h;
        CU(cuMemAddressReserve(&va, sz, 0, 0, 0));
        CU(cuMemCreate(&h, sz, &p, 0));
        CU(cuMemMap(va, sz, 0, h, 0));
        CU(cuMemSetAccess(va, sz, &a, 1));
        CU(cuMemUnmap(va, sz));
        CU(cuMemRelease(h));
        CU(cuMemAddressFree(va, sz));
        nvtxRangePop();
        if (gap_ms) {
            auto t = std::chrono::steady_clock::now() + std::chrono::milliseconds(gap_ms);
            while (std::chrono::steady_clock::now() < t) {}
        }
    }
    fprintf(stderr, "done: %zu runs\n", runs.size());
    CU(cuDevicePrimaryCtxRelease(dev));
    return 0;
}
