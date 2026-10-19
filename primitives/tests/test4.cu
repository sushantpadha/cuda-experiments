// test 4: remap is not atomic. a kernel keeps touching a chunk while the host remaps it under the same address
// one trial per process (a fault kills the context); tests/test4.sh runs many
// make test N=4 ARGS="64 2000 4"   (chunk MiB, kernel passes, host<->device round trips)
// exit: 0 OK, 1 CORRUPT (kernel finished, wrong data), 2 CRASH (sticky error), 3 setup failed
#include "pattern.cuh"
#include "vmem.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// every pass adds 1 to every word; volatile so each pass really loads and stores
__global__ void bump(volatile unsigned *p, size_t n, int passes) {
    for (int k = 0; k < passes; ++k)
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
            p[i] = p[i] + 1;
}

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? atoi(argv[1]) : 64;
    int passes = argc > 2 ? atoi(argv[2]) : 2000;
    int rounds = argc > 3 ? atoi(argv[3]) : 4;
    size_t n = (mib << 20) / sizeof(unsigned);

    vmem::Options o;
    o.verbose = false;
    vmem::Manager m(o);
    vmem::ChunkToken c;
    unsigned *p;
    cudaStream_t s;

    // ---- setup: chunk on device, word i = i + 1 ----
    try {
        CUdeviceptr va = m.reserve(mib << 20);
        c = m.create(mib << 20, vmem::Loc::Device);
        m.map(c, va);
        p = m.ptr<unsigned>(c);
        if (pattern::fill(p, n, 1) != cudaSuccess) throw vmem::Error("fill failed");
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    } catch (const vmem::Error &e) {
        printf("RESULT: SETUP %s\n", e.what());
        return 3;
    }

    // ---- race: kernel on a non-blocking stream, host remaps underneath ----
    bump<<<160, 256, 0, s>>>(p, n, passes);
    int done = 0;
    bool busy = true;
    const char *threw = nullptr;
    std::string what;
    try {
        for (; done < rounds; ++done) {
            m.remap(c, vmem::Loc::Host);
            m.remap(c, vmem::Loc::Device);
        }
        busy = cudaStreamQuery(s) == cudaErrorNotReady;   // did the remaps overlap the kernel at all
    } catch (const vmem::Error &e) {
        threw = "remap threw";
        what = e.what();
    }

    // ---- classify ----
    cudaError_t e = cudaStreamSynchronize(s);
    if (threw || e != cudaSuccess) {
        printf("RESULT: CRASH after %d round trip(s): sync -> %s (%s)%s%s\n", done, cudaGetErrorName(e),
               cudaGetErrorString(e), threw ? "; remap threw: " : "", what.c_str());
        return 2;
    }
    std::vector<unsigned> h(n);
    if ((e = cudaMemcpy(h.data(), p, mib << 20, cudaMemcpyDeviceToHost)) != cudaSuccess) {
        printf("RESULT: CRASH on readback: %s\n", cudaGetErrorName(e));
        return 2;
    }
    size_t bad = 0, first = n;
    for (size_t i = 0; i < n; ++i)
        if (h[i] != (unsigned)(i + 1 + passes)) bad += 1, first = first < i ? first : i;
    if (bad) {
        printf("RESULT: CORRUPT %zu of %zu words wrong; word %zu = %u, expected %u (lost %d of %d passes)\n", bad,
               n, first, h[first], (unsigned)(first + 1 + passes), (int)(first + 1 + passes - h[first]), passes);
        return 1;
    }
    printf("RESULT: OK %d round trip(s), kernel %s when they ended\n", done, busy ? "still running" : "already done");
    return 0;
}
