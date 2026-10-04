// Tracker 2B probe: per-launch working-set inference from kernel arguments.
// LD_PRELOAD shim. Hooks cudaMalloc/cudaFree to know allocations, and
// cudaLaunchKernel to scan each launch's parameters for pointers that land in
// a known allocation. Parameter layout comes from cuFuncGetParamInfo (no
// offline profiling). Params larger than 8 bytes (structs by value) are scanned
// at every 8-byte offset. Prints a summary at exit; set ARGSHIM_LOG=1 to print
// every launch.
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <mutex>
#include <chrono>
#include <unordered_map>

static void scan(const void *key, bool is_kernel, void **args);
namespace {
struct Alloc { uintptr_t base; size_t size; int id; long uses = 0; long last_launch = -1; };
std::map<uintptr_t, Alloc> g_allocs;           // keyed by base
std::mutex g_mu;
int g_next_id = 0;
long g_launches = 0, g_launches_with_hits = 0, g_ptr_hits = 0, g_cu_launches = 0;
double g_hook_ns = 0;
bool g_log = getenv("ARGSHIM_LOG") != nullptr;
struct Layout { std::vector<std::pair<size_t, size_t>> params; std::string name; };
std::unordered_map<const void *, Layout> g_layouts;

template <class F> F real(const char *sym) { return (F)dlsym(RTLD_NEXT, sym); }

Alloc *lookup(uintptr_t p) {        // allocation containing p, or null
    auto it = g_allocs.upper_bound(p);
    if (it == g_allocs.begin()) return nullptr;
    --it;
    return p < it->second.base + it->second.size ? &it->second : nullptr;
}

// CUDA 13 kernel stubs pass a cudaKernel_t (== CUkernel) instead of a host
// function pointer; read its layout with the cuKernel* calls.
const Layout &layout_for_kernel(cudaKernel_t k) {
    auto it = g_layouts.find((const void *)k);
    if (it != g_layouts.end()) return it->second;
    Layout L;
    const char *nm = nullptr; if (cuKernelGetName(&nm, (CUkernel)k) == CUDA_SUCCESS && nm) L.name = nm;
    for (size_t i = 0;; ++i) {
        size_t off, sz;
        if (cuKernelGetParamInfo((CUkernel)k, i, &off, &sz) != CUDA_SUCCESS) break;
        L.params.push_back({off, sz});
    }
    return g_layouts.emplace((const void *)k, std::move(L)).first->second;
}

const Layout &layout_for(const void *func) {
    auto it = g_layouts.find(func);
    if (it != g_layouts.end()) return it->second;
    Layout L;
    cudaFunction_t f;
    if (cudaGetFuncBySymbol(&f, func) == cudaSuccess) {
        const char *nm = nullptr; if (cuFuncGetName(&nm, (CUfunction)f) == CUDA_SUCCESS && nm) L.name = nm;
        for (size_t i = 0;; ++i) {
            size_t off, sz;
            if (cuFuncGetParamInfo((CUfunction)f, i, &off, &sz) != CUDA_SUCCESS) break;
            L.params.push_back({off, sz});
        }
    }
    return g_layouts.emplace(func, std::move(L)).first->second;
}

struct Report { ~Report() {
    fprintf(stderr, "[argshim] runtime launches=%ld with>=1 alloc hit=%ld ptr hits=%ld; avg hook cost=%.0f ns/launch; driver cuLaunchKernel calls seen=%ld\n",
            g_launches, g_launches_with_hits, g_ptr_hits, g_launches ? g_hook_ns / g_launches : 0.0, g_cu_launches);
    for (auto &[b, a] : g_allocs)
        fprintf(stderr, "[argshim]   alloc#%d size=%zu MiB uses=%ld last_launch=%ld\n", a.id, a.size >> 20, a.uses, a.last_launch);
} } g_report;
}  // namespace

extern "C" cudaError_t cudaMalloc(void **p, size_t size) {
    static auto f = real<cudaError_t (*)(void **, size_t)>("cudaMalloc");
    cudaError_t e = f(p, size);
    if (e == cudaSuccess) { std::lock_guard<std::mutex> l(g_mu); g_allocs[(uintptr_t)*p] = {(uintptr_t)*p, size, g_next_id++}; }
    return e;
}
extern "C" cudaError_t cudaFree(void *p) {
    static auto f = real<cudaError_t (*)(void *)>("cudaFree");
    { std::lock_guard<std::mutex> l(g_mu); g_allocs.erase((uintptr_t)p); }
    return f(p);
}
static void scan(const void *key, bool is_kernel, void **args) {
    auto t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> l(g_mu);
        const Layout &L = is_kernel ? layout_for_kernel((cudaKernel_t)key) : layout_for(key);
        std::set<int> hit;
        for (size_t i = 0; i < L.params.size(); ++i) {
            size_t sz = L.params[i].second;
            const char *arg = (const char *)args[i];
            for (size_t o = 0; o + 8 <= sz; o += 8) {
                uintptr_t v; memcpy(&v, arg + o, 8);
                if (Alloc *a = lookup(v)) { hit.insert(a->id); a->uses++; a->last_launch = g_launches; g_ptr_hits++; }
            }
        }
        if (!hit.empty()) g_launches_with_hits++;
        if (g_log) {
            fprintf(stderr, "[argshim] launch %ld %s params=%zu allocs={", g_launches, L.name.c_str(), L.params.size());
            for (int id : hit) fprintf(stderr, " %d", id);
            fprintf(stderr, " }\n");
        }
        g_launches++;
    }
    g_hook_ns += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
}

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 g, dim3 b, void **args, size_t shm, cudaStream_t s) {
    static auto f = real<cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t)>("cudaLaunchKernel");
    scan(func, false, args);
    return f(func, g, b, args, shm, s);
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t k, dim3 g, dim3 b, void **args, size_t shm, cudaStream_t s) {
    static auto f = real<cudaError_t (*)(cudaKernel_t, dim3, dim3, void **, size_t, cudaStream_t)>("__cudaLaunchKernel");
    scan((const void *)k, true, args);
    return f(k, g, b, args, shm, s);
}
extern "C" cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t k, dim3 g, dim3 b, void **args, size_t shm, cudaStream_t s) {
    static auto f = real<cudaError_t (*)(cudaKernel_t, dim3, dim3, void **, size_t, cudaStream_t)>("__cudaLaunchKernel_ptsz");
    scan((const void *)k, true, args);
    return f(k, g, b, args, shm, s);
}
// Driver-API launches (libraries that bypass cudaLaunchKernel). Only counted:
// many libraries resolve driver entry points through cuGetProcAddress, which
// LD_PRELOAD cannot see, so a low count here is expected.
extern "C" CUresult cuLaunchKernel(CUfunction fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                                   unsigned shm, CUstream s, void **params, void **extra) {
    static auto f = real<CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **)>("cuLaunchKernel");
    __atomic_add_fetch(&g_cu_launches, 1, __ATOMIC_RELAXED);
    return f(fn, gx, gy, gz, bx, by, bz, shm, s, params, extra);
}
