// Tracker 2B probe, take 2: same working-set inference as argshim.so, but
// injected through CUPTI (CUDA_INJECTION64_PATH) and attached to *driver* API
// callbacks, so launches from libraries that bypass the runtime (cuBLAS, ...)
// are seen too. No relink, no LD_PRELOAD.
//   allocations: cuMemAlloc_v2, cuMemAllocAsync(_ptsz), cuMemMap (VMM)
//   frees:       cuMemFree_v2, cuMemUnmap
//   launches:    cuLaunchKernel(_ptsz), cuLaunchKernelEx(_ptsz)
// Kernel params come from kernelParams (layout via cuFuncGetParamInfo), or from
// the packed `extra` buffer (scanned at 8-byte steps).
#include <cuda.h>
#include <cupti.h>
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

namespace {
struct Alloc { uintptr_t base; size_t size; int id; long uses = 0; long last_launch = -1; };
std::map<uintptr_t, Alloc> g_allocs;
std::mutex g_mu;
int g_next_id = 0;
long g_launches = 0, g_with_hits = 0, g_ptr_hits = 0, g_extra = 0;
double g_ns = 0;
bool g_log = getenv("ARGSHIM_LOG") != nullptr;
struct Layout { std::vector<std::pair<size_t, size_t>> params; std::string name; };
std::unordered_map<CUfunction, Layout> g_layouts;
bool g_noscan = getenv("ARGSHIM_NOSCAN") != nullptr;   // subscribe only: isolates CUPTI's own cost
std::map<std::string, long> g_names;   // kernel name -> launches

Alloc *lookup(uintptr_t p) {
    auto it = g_allocs.upper_bound(p);
    if (it == g_allocs.begin()) return nullptr;
    --it;
    return p < it->second.base + it->second.size ? &it->second : nullptr;
}
void add_alloc(uintptr_t b, size_t sz) { g_allocs[b] = {b, sz, g_next_id++}; }

void scan_launch(CUfunction f, void **kp, void **extra) {
    auto t0 = std::chrono::steady_clock::now();
    std::set<int> hit;
    auto touch = [&](const char *p, size_t sz) {
        for (size_t o = 0; o + 8 <= sz; o += 8) {
            uintptr_t v; memcpy(&v, p + o, 8);
            if (Alloc *a = lookup(v)) { hit.insert(a->id); a->uses++; a->last_launch = g_launches; g_ptr_hits++; }
        }
    };
    if (g_noscan) { g_launches++; return; }
    auto it = g_layouts.find(f);
    if (it == g_layouts.end()) {
        Layout L; size_t off, sz;
        for (size_t i = 0; cuFuncGetParamInfo(f, i, &off, &sz) == CUDA_SUCCESS; ++i) L.params.push_back({off, sz});
        if (L.params.empty())   // runtime may hand the driver a CUkernel in the CUfunction slot
            for (size_t i = 0; cuKernelGetParamInfo((CUkernel)f, i, &off, &sz) == CUDA_SUCCESS; ++i) L.params.push_back({off, sz});
        const char *nm = nullptr;
        if (cuFuncGetName(&nm, f) != CUDA_SUCCESS || !nm) { nm = nullptr; cuKernelGetName(&nm, (CUkernel)f); }
        L.name = nm ? nm : "?";
        it = g_layouts.emplace(f, std::move(L)).first;
    }
    if (kp) {
        for (size_t i = 0; i < it->second.params.size(); ++i) touch((const char *)kp[i], it->second.params[i].second);
    } else if (extra) {
        g_extra++;
        const char *buf = nullptr; size_t sz = 0;
        for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = (const char *)extra[i + 1];
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) sz = *(size_t *)extra[i + 1];
        }
        if (buf) touch(buf, sz);
    }
    const char *nm = it->second.name.c_str();
    g_names[it->second.name]++;
    if (!hit.empty()) g_with_hits++;
    if (g_log) { fprintf(stderr, "[cupti] launch %ld %.60s allocs={", g_launches, nm); for (int id : hit) fprintf(stderr, " %d", id); fprintf(stderr, " }\n"); }
    g_launches++;
    g_ns += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
}

void CUPTIAPI cb(void *, CUpti_CallbackDomain dom, CUpti_CallbackId id, const void *data) {
    if (dom != CUPTI_CB_DOMAIN_DRIVER_API) return;
    auto *ci = (const CUpti_CallbackData *)data;
    std::lock_guard<std::mutex> l(g_mu);
    if (ci->callbackSite == CUPTI_API_ENTER) {
        switch (id) {
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz: {
            auto *p = (const cuLaunchKernel_params *)ci->functionParams; scan_launch(p->f, p->kernelParams, p->extra); break; }
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx: case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
            auto *p = (const cuLaunchKernelEx_params *)ci->functionParams; scan_launch(p->f, p->kernelParams, p->extra); break; }
        case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2: g_allocs.erase((uintptr_t)((const cuMemFree_v2_params *)ci->functionParams)->dptr); break;
        case CUPTI_DRIVER_TRACE_CBID_cuMemUnmap: g_allocs.erase((uintptr_t)((const cuMemUnmap_params *)ci->functionParams)->ptr); break;
        default: break;
        }
    } else {   // API_EXIT: the allocation exists now
        switch (id) {
        case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2: { auto *p = (const cuMemAlloc_v2_params *)ci->functionParams; add_alloc(*p->dptr, p->bytesize); break; }
        case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync: case CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz: {
            auto *p = (const cuMemAllocAsync_params *)ci->functionParams; add_alloc(*p->dptr, p->bytesize); break; }
        case CUPTI_DRIVER_TRACE_CBID_cuMemMap: { auto *p = (const cuMemMap_params *)ci->functionParams; add_alloc(p->ptr, p->size); break; }
        default: break;
        }
    }
}

struct Report { ~Report() {
    fprintf(stderr, "[cupti] driver launches=%ld with>=1 alloc hit=%ld ptr hits=%ld via-extra=%ld; avg scan cost=%.0f ns/launch\n",
            g_launches, g_with_hits, g_ptr_hits, g_extra, g_launches ? g_ns / g_launches : 0.0);
    for (auto &[n, c] : g_names) fprintf(stderr, "[cupti]   kernel %-70.70s x%ld\n", n.c_str(), c);
    for (auto &[b, a] : g_allocs) fprintf(stderr, "[cupti]   alloc#%d size=%zu KiB uses=%ld last_launch=%ld\n", a.id, a.size >> 10, a.uses, a.last_launch);
} } g_report;
}  // namespace

extern "C" int InitializeInjection() {
    CUpti_SubscriberHandle sub;
    if (cuptiSubscribe(&sub, (CUpti_CallbackFunc)cb, nullptr) != CUPTI_SUCCESS) { fprintf(stderr, "[cupti] subscribe failed\n"); return 0; }
    CUpti_CallbackId ids[] = {CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2, CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync, CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuMemMap, CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2, CUPTI_DRIVER_TRACE_CBID_cuMemUnmap};
    for (auto id : ids) cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, id);
    return 1;
}
