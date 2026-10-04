// Tracker 2B probes for eviction signals that need no hardware counters.
//   frac  : kernel time vs. fraction of a buffer demoted to HOST_NUMA (is kernel
//           time a usable "pain" signal for a tenant whose data was demoted?)
//   hash  : GPU-side per-block hashing throughput, and whether a 1-byte change
//           is caught (dirty detection without dirty bits)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CU(c) do { CUresult e_ = (c); if (e_) { const char *n_; cuGetErrorName(e_, &n_); printf("%s:%d %s\n", __FILE__, __LINE__, n_); exit(1);} } while (0)
#define RT(c) do { cudaError_t e_ = (c); if (e_) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorName(e_)); exit(1);} } while (0)

static CUdevice dev; static CUcontext ctx;
static CUmemAllocationProp prop(bool host) {
    CUmemAllocationProp p{}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (host) { p.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; p.location.id = 0; }
    else { p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = 0; }
    return p;
}
static void access(CUdeviceptr va, size_t sz) {
    CUmemAccessDesc a{}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = 0;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CU(cuMemSetAccess(va, sz, &a, 1));
}

__global__ void stream_rw(float *p, size_t n) {   // read-modify-write, one pass
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = p[i] * 1.0001f + 1.f;
}
__global__ void reuse_rw(float *p, size_t n, int reps) {   // `reps` full streaming passes; NOT a cache-reuse test (512 MiB >> 24 MB L2)
    for (int r = 0; r < reps; ++r)
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = p[i] * 1.0001f + 1.f;
}

// 64-bit FNV-style mix per 2 MiB block; one CTA per block, warp-shuffle + shared reduce.
__global__ void hash_blocks(const unsigned long long *p, size_t words_per_block, unsigned long long *out) {
    const unsigned long long *b = p + blockIdx.x * words_per_block;
    unsigned long long h = 0;
    for (size_t i = threadIdx.x; i < words_per_block; i += blockDim.x) {
        unsigned long long v = b[i] ^ (i * 0x9E3779B97F4A7C15ull);
        v *= 0xff51afd7ed558ccdull; v ^= v >> 33; h += v;   // order-independent sum of mixed words
    }
    for (int o = 16; o; o >>= 1) h += __shfl_down_sync(0xffffffff, h, o);
    __shared__ unsigned long long s[32];
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = h;
    __syncthreads();
    if (threadIdx.x == 0) { unsigned long long t = 0; for (int w = 0; w < blockDim.x / 32; ++w) t += s[w]; out[blockIdx.x] = t; }
}

static float time_kernel(void (*launch)(float *, size_t, int), float *p, size_t n, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    launch(p, n, reps); RT(cudaDeviceSynchronize());
    cudaEventRecord(a); launch(p, n, reps); cudaEventRecord(b); RT(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); return ms;
}
static void l_stream(float *p, size_t n, int) { stream_rw<<<160, 256>>>(p, n); }
static void l_reuse(float *p, size_t n, int reps) { reuse_rw<<<160, 256>>>(p, n, reps); }

static void mode_frac() {
    const int CH = 16; const size_t CS = 32ull << 20, SZ = CH * CS; size_t n = SZ / 4;  // 512 MiB in 16 x 32 MiB chunks
    CUdeviceptr va; CU(cuMemAddressReserve(&va, SZ, 0, 0, 0));
    std::vector<CUmemGenericAllocationHandle> hd(CH), hh(CH);
    CUmemAllocationProp pd = prop(false), ph = prop(true);
    for (int i = 0; i < CH; ++i) { CU(cuMemCreate(&hd[i], CS, &pd, 0)); CU(cuMemCreate(&hh[i], CS, &ph, 0)); }
    printf("chunks_on_host  stream_ms  reuse8_ms   (512 MiB buffer, 16 x 32 MiB chunks)\n");
    for (int k = 0; k <= CH; k += 2) {
        for (int i = 0; i < CH; ++i) { CU(cuMemMap(va + i * CS, CS, 0, i < k ? hh[i] : hd[i], 0)); }
        access(va, SZ);
        float s = time_kernel(l_stream, (float *)va, n, 0), r = time_kernel(l_reuse, (float *)va, n, 8);
        printf("%2d/%d           %8.2f   %8.2f\n", k, CH, s, r);
        for (int i = 0; i < CH; ++i) CU(cuMemUnmap(va + i * CS, CS));
    }
}

static void mode_hash() {
    const size_t BS = 2ull << 20; const int NB = 512; const size_t SZ = BS * NB;   // 1 GiB, 512 blocks
    unsigned long long *p, *h0, *h1; RT(cudaMalloc(&p, SZ)); RT(cudaMalloc(&h0, NB * 8)); RT(cudaMalloc(&h1, NB * 8));
    RT(cudaMemset(p, 0x5a, SZ));
    size_t wpb = BS / 8;
    hash_blocks<<<NB, 512>>>(p, wpb, h0); RT(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < 5; ++i) hash_blocks<<<NB, 512>>>(p, wpb, h1); cudaEventRecord(b); RT(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= 5;
    printf("hash 1 GiB (512 x 2 MiB blocks): %.2f ms = %.1f GB/s\n", ms, SZ / (ms * 1e-3) / 1e9);
    // flip one byte in block 137 and one in block 400; expect exactly those to differ
    unsigned char x = 0x5b; RT(cudaMemcpy((char *)p + 137 * BS + 12345, &x, 1, cudaMemcpyHostToDevice));
    RT(cudaMemcpy((char *)p + 400 * BS + BS - 1, &x, 1, cudaMemcpyHostToDevice));
    hash_blocks<<<NB, 512>>>(p, wpb, h1); RT(cudaDeviceSynchronize());
    std::vector<unsigned long long> A(NB), B(NB); RT(cudaMemcpy(A.data(), h0, NB * 8, cudaMemcpyDeviceToHost)); RT(cudaMemcpy(B.data(), h1, NB * 8, cudaMemcpyDeviceToHost));
    printf("blocks whose hash changed after 2 one-byte writes:");
    for (int i = 0; i < NB; ++i) if (A[i] != B[i]) printf(" %d", i);
    printf("   (expected: 137 400)\n");
    // for scale: copying the same 1 GiB to host
    void *hbuf; RT(cudaMallocHost(&hbuf, SZ));
    cudaEventRecord(a); RT(cudaMemcpy(hbuf, p, SZ, cudaMemcpyDeviceToHost)); cudaEventRecord(b); RT(cudaEventSynchronize(b));
    cudaEventElapsedTime(&ms, a, b); printf("copy 1 GiB D2H (pinned): %.2f ms = %.1f GB/s\n", ms, SZ / (ms * 1e-3) / 1e9);
}

int main(int argc, char **argv) {
    CU(cuInit(0)); CU(cuDeviceGet(&dev, 0)); CU(cuDevicePrimaryCtxRetain(&ctx, dev)); CU(cuCtxSetCurrent(ctx));
    std::string m = argc > 1 ? argv[1] : "";
    if (m == "frac") mode_frac(); else if (m == "hash") mode_hash(); else { printf("usage: signals frac|hash\n"); return 1; }
}
