// Tracker 2A probes: can several processes / contexts run kernels at the same
// time on this GPU, and do VMM calls work in each setting?
//
// Modes (see README.md for the exact runs):
//   work  <iters> <blocks> [green_sms]   fixed-work kernel, optionally in a green context
//   green <iters> <blocks> <sms_a>       one process, two green contexts on disjoint SMs
//   vmm                                  device + HOST_NUMA backing, kernel R/W, remap, bandwidth
//   share                                fork: export a VMM handle as a POSIX fd, child maps and R/Ws it
//   hold  <seconds>                      create a context and sleep (for per-context memory overhead)
//
// Every kernel records %globaltimer (device-wide ns clock, comparable across
// processes) and %smid per block, so overlap between processes is read off the
// device clock, not inferred from wall time.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <cmath>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>

#define CU(call) do { CUresult e_ = (call); if (e_ != CUDA_SUCCESS) { const char *n_; \
    cuGetErrorName(e_, &n_); fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, n_); exit(1);} } while (0)
#define RT(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorName(e_)); exit(1);} } while (0)

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

struct BlockRec { unsigned long long t0, t1; unsigned sm; float sink; };

// Dependent FMA chain: fixed work, independent of wall time, so time slicing
// shows up as a longer kernel instead of being hidden.
__global__ void work_kernel(BlockRec *rec, long iters) {
    unsigned long long t0 = gtimer();
    float a = threadIdx.x * 1e-3f, b = 1.0000001f;
    for (long i = 0; i < iters; ++i) a = fmaf(a, b, 1e-7f);
    __syncthreads();
    if (threadIdx.x == 0) { rec[blockIdx.x] = {t0, gtimer(), smid(), a}; }
}

__global__ void fill(unsigned *p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (unsigned)i * 2654435761u + seed;
}
__global__ void check(const unsigned *p, size_t n, unsigned seed, unsigned long long *bad) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != (unsigned)i * 2654435761u + seed) atomicAdd(bad, 1ull);
}
// Coalesced streaming read (uint4 per thread); result folded into out to keep loads live.
__global__ void readbw(const uint4 *p, size_t n, unsigned *out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) *out = acc;
}
// Random 4-byte reads: one per thread per step, pseudo-random index.
__global__ void readrand(const unsigned *p, size_t n, int steps, unsigned *out) {
    unsigned acc = 0, x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    for (int s = 0; s < steps; ++s) { x = x * 1664525u + 1013904223u; acc ^= p[x % n]; }
    if (acc == 0x12345678u) *out = acc;
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static CUcontext g_primary;
static CUdevice g_dev;
static void init() {
    CU(cuInit(0)); CU(cuDeviceGet(&g_dev, 0));
    CU(cuDevicePrimaryCtxRetain(&g_primary, g_dev)); CU(cuCtxSetCurrent(g_primary));
}

static void report(const char *tag, std::vector<BlockRec> &r, double wall) {
    unsigned long long t0 = ~0ull, t1 = 0; std::set<unsigned> sms;
    for (auto &b : r) { t0 = std::min(t0, b.t0); t1 = std::max(t1, b.t1); sms.insert(b.sm); }
    printf("%s pid=%d blocks=%zu dev_start_ns=%llu dev_end_ns=%llu dev_ms=%.2f wall_ms=%.2f sms={",
           tag, getpid(), r.size(), t0, t1, (t1 - t0) / 1e6, wall * 1e3);
    bool first = true; for (unsigned s : sms) { printf(first ? "%u" : ",%u", s); first = false; }
    printf("}\n"); fflush(stdout);
}

// Make a green context with `sms` SMs; returns its stream.
// sms < 0: split off |sms| SMs and use the *remaining* group instead (disjoint
// from what another process asking for +|sms| gets).
static CUstream make_green(int sms, CUgreenCtx *out_gc, CUdevResource *remaining) {
    CUdevResource all{}, part{}, rem{};
    CU(cuDeviceGetDevResource(g_dev, &all, CU_DEV_RESOURCE_TYPE_SM));
    unsigned n = 1; bool use_rem = sms < 0; if (use_rem) sms = -sms;
    CU(cuDevSmResourceSplitByCount(&part, &n, &all, &rem, 0, sms));
    CUdevResourceDesc d; CU(cuDevResourceGenerateDesc(&d, use_rem ? &rem : &part, 1));
    CUgreenCtx gc; CU(cuGreenCtxCreate(&gc, d, g_dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUstream s; CU(cuGreenCtxStreamCreate(&s, gc, CU_STREAM_NON_BLOCKING, 0));
    printf("green ctx: requested %d SMs, got %u, remaining %u\n", sms, part.sm.smCount, rem.sm.smCount);
    *out_gc = gc; if (remaining) *remaining = rem;
    return s;
}

static void mode_work(long iters, int blocks, int green_sms) {
    init();
    CUstream s = 0; CUgreenCtx gc = nullptr;
    if (green_sms != 0) {
        s = make_green(green_sms, &gc, nullptr);
        CUcontext c; CU(cuCtxFromGreenCtx(&c, gc)); CU(cuCtxSetCurrent(c));
    }
    BlockRec *d; RT(cudaMalloc(&d, blocks * sizeof(BlockRec)));
    work_kernel<<<blocks, 128, 0, (cudaStream_t)s>>>(d, 1000); RT(cudaStreamSynchronize((cudaStream_t)s)); // warm-up
    // Line up processes started together: wait for the next whole wall-clock 500 ms.
    double t = now_s(); usleep((useconds_t)((0.5 - fmod(t, 0.5)) * 1e6));
    double w0 = now_s();
    work_kernel<<<blocks, 128, 0, (cudaStream_t)s>>>(d, iters);
    RT(cudaStreamSynchronize((cudaStream_t)s));
    double w = now_s() - w0;
    std::vector<BlockRec> r(blocks); RT(cudaMemcpy(r.data(), d, blocks * sizeof(BlockRec), cudaMemcpyDeviceToHost));
    report(green_sms != 0 ? "work[green]" : "work", r, w);
}

static void mode_green(long iters, int blocks, int sms_a) {
    init();
    CUdevResource all{}; CU(cuDeviceGetDevResource(g_dev, &all, CU_DEV_RESOURCE_TYPE_SM));
    CUdevResource part{}, rem{}; unsigned n = 1;
    CU(cuDevSmResourceSplitByCount(&part, &n, &all, &rem, 0, sms_a));
    CUdevResourceDesc da, db; CU(cuDevResourceGenerateDesc(&da, &part, 1)); CU(cuDevResourceGenerateDesc(&db, &rem, 1));
    CUgreenCtx ga, gb; CU(cuGreenCtxCreate(&ga, da, g_dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CU(cuGreenCtxCreate(&gb, db, g_dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CUstream sa, sb; CU(cuGreenCtxStreamCreate(&sa, ga, CU_STREAM_NON_BLOCKING, 0)); CU(cuGreenCtxStreamCreate(&sb, gb, CU_STREAM_NON_BLOCKING, 0));
    printf("green A: %u SMs, green B: %u SMs\n", part.sm.smCount, rem.sm.smCount);
    BlockRec *d; RT(cudaMalloc(&d, 2 * blocks * sizeof(BlockRec)));
    work_kernel<<<blocks, 128, 0, (cudaStream_t)sa>>>(d, 1000); work_kernel<<<blocks, 128, 0, (cudaStream_t)sb>>>(d + blocks, 1000);
    RT(cudaDeviceSynchronize());
    double w0 = now_s();
    work_kernel<<<blocks, 128, 0, (cudaStream_t)sa>>>(d, iters);
    work_kernel<<<blocks, 128, 0, (cudaStream_t)sb>>>(d + blocks, iters);
    RT(cudaStreamSynchronize((cudaStream_t)sa)); RT(cudaStreamSynchronize((cudaStream_t)sb));
    double w = now_s() - w0;
    std::vector<BlockRec> r(2 * blocks); RT(cudaMemcpy(r.data(), d, r.size() * sizeof(BlockRec), cudaMemcpyDeviceToHost));
    std::vector<BlockRec> ra(r.begin(), r.begin() + blocks), rb(r.begin() + blocks, r.end());
    report("green-A", ra, w); report("green-B", rb, w);
}

// ---- VMM helpers ----
static CUmemAllocationProp prop_for(bool host) {
    CUmemAllocationProp p{}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (host) { int numa = 0; cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, g_dev);
        p.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA; p.location.id = numa < 0 ? 0 : numa; }
    else { p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = 0; }
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    return p;
}
static void grant_device(CUdeviceptr va, size_t sz) {
    CUmemAccessDesc a{}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = 0;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CU(cuMemSetAccess(va, sz, &a, 1));
}
static unsigned long long verify(CUdeviceptr va, size_t n, unsigned seed) {
    unsigned long long *bad; RT(cudaMallocManaged(&bad, 8)); *bad = 0;
    check<<<160, 256>>>((const unsigned *)va, n, seed, bad); RT(cudaDeviceSynchronize());
    unsigned long long b = *bad; RT(cudaFree(bad)); return b;
}
static double bw_seq(CUdeviceptr va, size_t bytes) {
    unsigned *o; RT(cudaMalloc(&o, 4));
    readbw<<<160, 256>>>((const uint4 *)va, bytes / 16, o); RT(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < 5; ++i) readbw<<<160, 256>>>((const uint4 *)va, bytes / 16, o); cudaEventRecord(b);
    RT(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); RT(cudaFree(o));
    return 5.0 * bytes / (ms * 1e-3) / 1e9;
}
static double rand_rate(CUdeviceptr va, size_t bytes) {   // million random 4-byte reads per second
    unsigned *o; RT(cudaMalloc(&o, 4)); int steps = 64, blocks = 160, thr = 256;
    readrand<<<blocks, thr>>>((const unsigned *)va, bytes / 4, 4, o); RT(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); readrand<<<blocks, thr>>>((const unsigned *)va, bytes / 4, steps, o); cudaEventRecord(b);
    RT(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); RT(cudaFree(o));
    return (double)blocks * thr * steps / (ms * 1e-3) / 1e6;
}

static void mode_vmm() {
    init();
    CUmemAllocationProp pd = prop_for(false), ph = prop_for(true);
    size_t gd = 0, gh = 0;
    CU(cuMemGetAllocationGranularity(&gd, &pd, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CU(cuMemGetAllocationGranularity(&gh, &ph, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    int numa_vmm = -1; cuDeviceGetAttribute(&numa_vmm, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, g_dev);
    printf("granularity device=%zu host_numa=%zu HOST_NUMA_VMM_SUPPORTED=%d\n", gd, gh, numa_vmm);
    const size_t SZ = 256ull << 20; size_t n = SZ / 4;
    CUdeviceptr va; CU(cuMemAddressReserve(&va, SZ, 0, 0, 0));
    CUmemGenericAllocationHandle hd, hh;
    CU(cuMemCreate(&hd, SZ, &pd, 0)); CU(cuMemMap(va, SZ, 0, hd, 0)); grant_device(va, SZ);
    fill<<<160, 256>>>((unsigned *)va, n, 7); RT(cudaDeviceSynchronize());
    printf("device-backed: verify bad=%llu  seq_read=%.1f GB/s  rand_read=%.1f M/s\n", verify(va, n, 7), bw_seq(va, SZ), rand_rate(va, SZ));
    // Remap the same VA to host memory, keeping contents (copy via a scratch VA).
    CU(cuMemCreate(&hh, SZ, &ph, 0));
    CUdeviceptr tmp; CU(cuMemAddressReserve(&tmp, SZ, 0, 0, 0)); CU(cuMemMap(tmp, SZ, 0, hh, 0)); grant_device(tmp, SZ);
    double t0 = now_s(); CU(cuMemcpyDtoD(tmp, va, SZ)); CU(cuCtxSynchronize()); double tc = now_s() - t0;
    t0 = now_s(); CU(cuMemUnmap(va, SZ)); CU(cuMemRelease(hd)); CU(cuMemMap(va, SZ, 0, hh, 0)); grant_device(va, SZ); double tm = now_s() - t0;
    CU(cuMemUnmap(tmp, SZ)); CU(cuMemAddressFree(tmp, SZ));
    printf("remap dev->host 256MiB: copy %.1f ms (%.1f GB/s), unmap+release+map+access %.3f ms\n", tc * 1e3, SZ / tc / 1e9, tm * 1e3);
    printf("host-backed (same VA): verify bad=%llu  seq_read=%.1f GB/s  rand_read=%.1f M/s\n", verify(va, n, 7), bw_seq(va, SZ), rand_rate(va, SZ));
    fill<<<160, 256>>>((unsigned *)va, n, 9); RT(cudaDeviceSynchronize());
    printf("host-backed kernel write then verify bad=%llu\n", verify(va, n, 9));
    CU(cuMemUnmap(va, SZ)); CU(cuMemRelease(hh)); CU(cuMemAddressFree(va, SZ));
}

// fd passing over a unix socketpair (SCM_RIGHTS)
static void send_fd(int sock, int fd) {
    char c = 0; iovec io{&c, 1}; char buf[CMSG_SPACE(sizeof(int))]{}; msghdr m{}; m.msg_iov = &io; m.msg_iovlen = 1;
    m.msg_control = buf; m.msg_controllen = sizeof buf; cmsghdr *cm = CMSG_FIRSTHDR(&m);
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int)); memcpy(CMSG_DATA(cm), &fd, sizeof fd);
    if (sendmsg(sock, &m, 0) < 0) { perror("sendmsg"); exit(1); }
}
static int recv_fd(int sock) {
    char c; iovec io{&c, 1}; char buf[CMSG_SPACE(sizeof(int))]{}; msghdr m{}; m.msg_iov = &io; m.msg_iovlen = 1;
    m.msg_control = buf; m.msg_controllen = sizeof buf; if (recvmsg(sock, &m, 0) <= 0) { perror("recvmsg"); exit(1); }
    int fd; memcpy(&fd, CMSG_DATA(CMSG_FIRSTHDR(&m)), sizeof fd); return fd;
}

static void share_one(bool host) {
    int sv[2]; socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    const size_t SZ = 64ull << 20; size_t n = SZ / 4;
    pid_t pid = fork();
    if (pid == 0) {   // child: import, verify owner's data, write new pattern
        close(sv[0]); init();
        int fd = recv_fd(sv[1]);
        CUmemGenericAllocationHandle h; CU(cuMemImportFromShareableHandle(&h, (void *)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        CUdeviceptr va; CU(cuMemAddressReserve(&va, SZ, 0, 0, 0)); CU(cuMemMap(va, SZ, 0, h, 0)); grant_device(va, SZ);
        printf("  [child pid=%d] imported %s handle, verify owner data bad=%llu\n", getpid(), host ? "HOST_NUMA" : "DEVICE", verify(va, n, 11));
        fill<<<160, 256>>>((unsigned *)va, n, 13); RT(cudaDeviceSynchronize());
        char ok = 1; write(sv[1], &ok, 1);
        CU(cuMemUnmap(va, SZ)); CU(cuMemRelease(h)); CU(cuMemAddressFree(va, SZ)); _exit(0);
    }
    close(sv[1]); init();
    CUmemAllocationProp p = prop_for(host);
    CUmemGenericAllocationHandle h; CU(cuMemCreate(&h, SZ, &p, 0));
    CUdeviceptr va; CU(cuMemAddressReserve(&va, SZ, 0, 0, 0)); CU(cuMemMap(va, SZ, 0, h, 0)); grant_device(va, SZ);
    fill<<<160, 256>>>((unsigned *)va, n, 11); RT(cudaDeviceSynchronize());
    int fd; CU(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    send_fd(sv[0], fd);
    char ok; read(sv[0], &ok, 1);
    printf("  [owner pid=%d] after child wrote: verify child data bad=%llu\n", getpid(), verify(va, n, 13));
    int st; waitpid(pid, &st, 0);
    printf("share %s: child exit status %d\n", host ? "HOST_NUMA" : "DEVICE", WEXITSTATUS(st));
    CU(cuMemUnmap(va, SZ)); CU(cuMemRelease(h)); CU(cuMemAddressFree(va, SZ));
}
static void mode_share() {
    // Each side calls cuInit after fork; a parent that initialised CUDA must not fork-and-use.
    pid_t p = fork(); if (p == 0) { share_one(false); _exit(0); } int st; waitpid(p, &st, 0);
    p = fork(); if (p == 0) { share_one(true); _exit(0); } waitpid(p, &st, 0);
}

// Does remapping buffer Y wait for an unrelated kernel that is still running on
// buffer X in the same process? Times each VMM step while the kernel runs.
static void mode_remapbusy(long iters) {
    init();
    CUmemAllocationProp pd = prop_for(false), ph = prop_for(true);
    const size_t SZ = 64ull << 20; size_t n = SZ / 4;
    BlockRec *rec; RT(cudaMalloc(&rec, 40 * sizeof(BlockRec)));
    cudaStream_t ks; RT(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking));
    CUstream cs; CU(cuStreamCreate(&cs, CU_STREAM_NON_BLOCKING));
    CUdeviceptr va; CU(cuMemAddressReserve(&va, SZ, 0, 0, 0));
    CUmemGenericAllocationHandle hd, hh; CU(cuMemCreate(&hd, SZ, &pd, 0)); CU(cuMemMap(va, SZ, 0, hd, 0)); grant_device(va, SZ);
    fill<<<160, 256>>>((unsigned *)va, n, 21); RT(cudaDeviceSynchronize());
    work_kernel<<<4, 128, 0, ks>>>(rec, 1000); RT(cudaStreamSynchronize(ks));
    for (int busy = 0; busy < 2; ++busy) {
        double tk = now_s();
        if (busy) work_kernel<<<4, 128, 0, ks>>>(rec, iters);   // unrelated kernel, other buffer
        double t0 = now_s(); CU(cuMemCreate(&hh, SZ, &ph, 0)); double t1 = now_s();
        CUdeviceptr tmp; CU(cuMemAddressReserve(&tmp, SZ, 0, 0, 0)); CU(cuMemMap(tmp, SZ, 0, hh, 0)); grant_device(tmp, SZ); double t2 = now_s();
        CU(cuMemcpyDtoDAsync(tmp, va, SZ, cs)); CU(cuStreamSynchronize(cs)); double t3 = now_s();
        CU(cuMemUnmap(va, SZ)); double t4 = now_s();
        CU(cuMemRelease(hd)); CU(cuMemMap(va, SZ, 0, hh, 0)); grant_device(va, SZ); double t5 = now_s();
        CU(cuMemUnmap(tmp, SZ)); CU(cuMemAddressFree(tmp, SZ));
        int kdone = cudaStreamQuery(ks) == cudaSuccess; RT(cudaStreamSynchronize(ks)); double tk1 = now_s();
        printf("%s: create %.2f ms | map scratch %.2f | copy64MiB %.2f | unmap %.2f | release+map+access %.2f | kernel_done_before_end=%d kernel_wall %.1f ms\n",
               busy ? "BUSY (kernel running)" : "IDLE", (t1-t0)*1e3, (t2-t1)*1e3, (t3-t2)*1e3, (t4-t3)*1e3, (t5-t4)*1e3, kdone, (tk1-tk)*1e3);
        printf("  verify after remap bad=%llu\n", verify(va, n, 21));
        // swap back to a device handle for the next round
        CU(cuMemCreate(&hd, SZ, &pd, 0)); CU(cuMemAddressReserve(&tmp, SZ, 0, 0, 0)); CU(cuMemMap(tmp, SZ, 0, hd, 0)); grant_device(tmp, SZ);
        CU(cuMemcpyDtoD(tmp, va, SZ)); CU(cuCtxSynchronize()); CU(cuMemUnmap(va, SZ)); CU(cuMemRelease(hh));
        CU(cuMemMap(va, SZ, 0, hd, 0)); grant_device(va, SZ); CU(cuMemUnmap(tmp, SZ)); CU(cuMemAddressFree(tmp, SZ));
    }
}

// Touch a reserved-but-unmapped VMM range from a kernel (the crash the host
// fallback is meant to avoid). Used to see what happens to *other* MPS clients.
static void mode_fault() {
    init(); const size_t SZ = 2ull << 20; CUdeviceptr va; CU(cuMemAddressReserve(&va, SZ, 0, 0, 0));
    fill<<<1, 32>>>((unsigned *)va, 32, 1);
    cudaError_t e = cudaDeviceSynchronize();
    printf("fault pid=%d: kernel on unmapped VMM range -> %s\n", getpid(), cudaGetErrorName(e));
}

// Remap a buffer while a kernel is reading it: is there a crash window between
// cuMemUnmap and cuMemMap?
__global__ void spinread(const unsigned *p, size_t n, long rounds, unsigned *out) {
    unsigned acc = 0;
    for (long r = 0; r < rounds; ++r)
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) acc += p[i];
    *out = acc;
}
static void mode_gaprace() {
    init(); CUmemAllocationProp pd = prop_for(false), ph = prop_for(true);
    const size_t SZ = 2ull << 20; size_t n = SZ / 4;
    CUdeviceptr va; CU(cuMemAddressReserve(&va, SZ, 0, 0, 0));
    CUmemGenericAllocationHandle hd, hh; CU(cuMemCreate(&hd, SZ, &pd, 0)); CU(cuMemMap(va, SZ, 0, hd, 0)); grant_device(va, SZ);
    CU(cuMemCreate(&hh, SZ, &ph, 0));
    unsigned *o; RT(cudaMalloc(&o, 4));
    spinread<<<1, 256>>>((const unsigned *)va, n, 20000, o);    // ~1 s of reads on va
    usleep(200000);
    CU(cuMemUnmap(va, SZ)); CU(cuMemMap(va, SZ, 0, hh, 0)); grant_device(va, SZ);   // swap backing under it
    cudaError_t e = cudaDeviceSynchronize();
    printf("gaprace: kernel reading during unmap/map -> %s\n", cudaGetErrorName(e));
}

// HOST_NUMA-only allocation (does the MPS device-memory cap count host memory?)
static void mode_hostalloc(int mib) {
    init(); CUmemAllocationProp ph = prop_for(true); size_t sz = (size_t)mib << 20;
    CUmemGenericAllocationHandle h; CUresult r = cuMemCreate(&h, sz, &ph, 0); const char *n; cuGetErrorName(r, &n);
    printf("hostalloc %d MiB HOST_NUMA -> %s\n", mib, n);
}
// Hold a context idle, then launch: does a co-tenant's fault in between break us?
static void mode_idlecheck(int secs) {
    init(); BlockRec *d; RT(cudaMalloc(&d, 4 * sizeof(BlockRec)));
    work_kernel<<<4, 128>>>(d, 1000); RT(cudaDeviceSynchronize());
    printf("idlecheck pid=%d: ready, idle for %d s\n", getpid(), secs); sleep(secs);
    work_kernel<<<4, 128>>>(d, 1000); cudaError_t e = cudaDeviceSynchronize();
    printf("idlecheck pid=%d: launch after idle -> %s\n", getpid(), cudaGetErrorName(e));
}

static void mode_hold(int secs) {
    init(); size_t f, t; CU(cuMemGetInfo(&f, &t));
    printf("hold pid=%d free=%zu MiB total=%zu MiB\n", getpid(), f >> 20, t >> 20); fflush(stdout);
    sleep(secs);
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s work|green|vmm|share|hold ...\n", argv[0]); return 1; }
    std::string m = argv[1];
    if (m == "work")  mode_work(atol(argv[2]), atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 0);
    else if (m == "green") mode_green(atol(argv[2]), atoi(argv[3]), atoi(argv[4]));
    else if (m == "vmm")   mode_vmm();
    else if (m == "share") mode_share();
    else if (m == "remapbusy") mode_remapbusy(atol(argv[2]));
    else if (m == "fault") mode_fault();
    else if (m == "gaprace") mode_gaprace();
    else if (m == "hostalloc") mode_hostalloc(atoi(argv[2]));
    else if (m == "idlecheck") mode_idlecheck(atoi(argv[2]));
    else if (m == "hold")  mode_hold(atoi(argv[2]));
    else { fprintf(stderr, "unknown mode\n"); return 1; }
    return 0;
}
