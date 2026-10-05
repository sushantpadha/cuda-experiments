// libvmem v0 primitives: blocking, error-checked wrapper over CUDA VMM
// per-process utility for exercising VMM; one Manager per process, not shared across processes
//
//   vmem::Manager m;
//   CUdeviceptr va = m.reserve(64 << 20);
//   vmem::ChunkToken c = m.create(32 << 20, vmem::Loc::Device);
//   m.map(c, va);
//   kernel<<<g, b>>>(m.ptr<float>(c));
//   m.remap(c, vmem::Loc::Host);
//   m.unmap(c); m.release(c); m.free(va, 64 << 20);
//
// single-threaded. never unmap/remap a chunk a kernel may be touching
#pragma once

#include <cuda.h>

#include <cstdarg>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

namespace vmem {

// ---- basics ----

enum class Loc { Device, Host };   // host = pinned host memory (HOST_NUMA)

inline const char *loc_name(Loc l) { return l == Loc::Device ? "device" : "host"; }

struct Error : std::runtime_error {
    CUresult code;
    Error(const std::string &what, CUresult c = CUDA_SUCCESS) : std::runtime_error(what), code(c) {}
};

inline std::string fmt(const char *f, ...) {
    char buf[512];
    va_list ap; va_start(ap, f); vsnprintf(buf, sizeof buf, f, ap); va_end(ap);
    return buf;
}

// throwing CU_CHECK
#define VMEM_CU(call)                                                                    \
    do {                                                                                 \
        CUresult e_ = (call);                                                            \
        if (e_ != CUDA_SUCCESS) {                                                        \
            const char *n_ = "?", *s_ = "?";                                             \
            cuGetErrorName(e_, &n_); cuGetErrorString(e_, &s_);                          \
            throw ::vmem::Error(::vmem::fmt("%s:%d %s -> %s (%s)", __FILE__, __LINE__,   \
                                            #call, n_, s_), e_);                         \
        }                                                                                \
    } while (0)

#define VMEM_REQUIRE(cond, ...) \
    do { if (!(cond)) throw ::vmem::Error(::vmem::fmt(__VA_ARGS__)); } while (0)

// ---- what callers see ----

// this is what callers hold for a chunk
struct ChunkToken {
    unsigned owner = 0;   // manager serial, 0 = invalid
    int id = -1;          // never reused
    bool valid() const { return owner != 0; }
    bool operator==(const ChunkToken &o) const { return owner == o.owner && id == o.id; }
    bool operator!=(const ChunkToken &o) const { return !(*this == o); }
};

// read-only copy of a chunk's state
struct ChunkInfo {
    int id;
    size_t size;
    Loc loc;
    CUdeviceptr va;       // 0 = unmapped
    bool mapped() const { return va != 0; }
};

struct Options {
    int device = 0;
    int host_numa = -1;    // -1 = node closest to the device (else 0)
    bool verbose = true;   // one line per call
    bool debug = false;    // dump state after every call
};

// ---- manager (per process) ----

class Manager {
    // one cuMemCreate handle
    struct Chunk {
        int id;
        size_t size;
        Loc loc;
        CUmemGenericAllocationHandle handle;
        CUdeviceptr va = 0;
        bool mapped() const { return va != 0; }
    };

public:
    explicit Manager(Options o = {}) : opt_(o), serial_(next_serial()) {
        VMEM_CU(cuInit(0));
        VMEM_CU(cuDeviceGet(&dev_, opt_.device));
        VMEM_CU(cuDevicePrimaryCtxRetain(&ctx_, dev_));
        VMEM_CU(cuCtxSetCurrent(ctx_));

        char name[256] = {};
        VMEM_CU(cuDeviceGetName(name, sizeof name, dev_));
        int vmm = 0, host_vmm = 0, numa = -1;
        VMEM_CU(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev_));
        VMEM_CU(cuDeviceGetAttribute(&host_vmm, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev_));
        VMEM_CU(cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev_));
        VMEM_REQUIRE(vmm, "device %d does not support VMM", opt_.device);
        VMEM_REQUIRE(host_vmm, "device %d does not support HOST_NUMA VMM allocations", opt_.device);
        numa_ = opt_.host_numa >= 0 ? opt_.host_numa : (numa < 0 ? 0 : numa);

        size_t gd = 0, gh = 0;
        CUmemAllocationProp pd = prop(Loc::Device), ph = prop(Loc::Host);
        VMEM_CU(cuMemGetAllocationGranularity(&gd, &pd, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        VMEM_CU(cuMemGetAllocationGranularity(&gh, &ph, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        gran_ = gd > gh ? gd : gh;   // same for both, so remap keeps sizes

        // blocking stream: remap copy waits for the default stream only
        VMEM_CU(cuStreamCreate(&copy_stream_, CU_STREAM_DEFAULT));

        size_t fr = 0, tot = 0;
        VMEM_CU(cuMemGetInfo(&fr, &tot));
        log("init: device %d \"%s\", host NUMA node %d, granularity %zu KiB, VRAM free %zu / %zu MiB",
            opt_.device, name, numa_, gran_ >> 10, fr >> 20, tot >> 20);
        dump();
    }

    ~Manager() {
        // best effort, never throws; failures are only printed
        size_t bad = 0;
        for (auto &[id, c] : chunks_) {
            if (c.mapped()) bad += !ok(cuMemUnmap(c.va, c.size), "cuMemUnmap", id);
            bad += !ok(cuMemRelease(c.handle), "cuMemRelease", id);
        }
        for (auto &[va, sz] : reservations_) bad += !ok(cuMemAddressFree(va, sz), "cuMemAddressFree", -1);
        if (copy_stream_) bad += !ok(cuStreamDestroy(copy_stream_), "cuStreamDestroy", -1);
        log("shutdown: released %zu chunk(s), freed %zu reservation(s), %zu failure(s)", chunks_.size(),
            reservations_.size(), bad);
        ok(cuDevicePrimaryCtxRelease(dev_), "cuDevicePrimaryCtxRelease", -1);
    }

    Manager(const Manager &) = delete;
    Manager &operator=(const Manager &) = delete;

    size_t granularity() const { return gran_; }

    // ---- queries ----

    ChunkInfo info(ChunkToken t) const {
        const Chunk &c = get(t, "info");
        return {c.id, c.size, c.loc, c.va};
    }

    // mapped address, same across remap
    CUdeviceptr va(ChunkToken t) const {
        const Chunk &c = get(t, "va");
        VMEM_REQUIRE(c.mapped(), "va: chunk#%d is not mapped", c.id);
        return c.va;
    }

    // va() typed for kernel args
    template <class T> T *ptr(ChunkToken t) const { return reinterpret_cast<T *>(va(t)); }

    // residency
    Loc loc(ChunkToken t) const { return get(t, "loc").loc; }
    bool on_device(ChunkToken t) const { return loc(t) == Loc::Device; }

    // ---- primitives ----

    // virtual range only, no memory behind it
    CUdeviceptr reserve(size_t size) {
        size_t sz = round_up(size);
        CUdeviceptr va = 0;
        VMEM_CU(cuMemAddressReserve(&va, sz, gran_, 0, 0));
        reservations_[va] = sz;
        log("reserve(%zu) -> va 0x%llx, %zu MiB", size, (unsigned long long)va, sz >> 20);
        dump();
        return va;
    }

    // physical memory on device or pinned host
    ChunkToken create(size_t size, Loc loc) {
        VMEM_REQUIRE(size > 0, "create: size must be > 0");
        size_t sz = round_up(size);
        CUmemAllocationProp p = prop(loc);
        CUmemGenericAllocationHandle h;
        VMEM_CU(cuMemCreate(&h, sz, &p, 0));
        int id = next_id_++;
        chunks_[id] = Chunk{id, sz, loc, h};
        log("create(%zu, %s) -> chunk#%d, %zu MiB", size, loc_name(loc), id, sz >> 20);
        dump();
        return {serial_, id};
    }

    // map inside a reservation, device gets read/write
    CUdeviceptr map(ChunkToken t, CUdeviceptr va) {
        Chunk *c = &get(t, "map");
        VMEM_REQUIRE(!c->mapped(), "map: chunk#%d is already mapped at 0x%llx", c->id, (unsigned long long)c->va);
        VMEM_REQUIRE(va % gran_ == 0, "map: va 0x%llx is not %zu-aligned", (unsigned long long)va, gran_);
        VMEM_REQUIRE(inside_reservation(va, c->size), "map: [0x%llx, +%zu) is not inside one reservation",
                     (unsigned long long)va, c->size);
        // ! extremely optional, overlap check
        // commented?
        // for (auto &[id, o] : chunks_)
        //     VMEM_REQUIRE(!(o.mapped() && va < o.va + o.size && o.va < va + c->size),
        //                  "map: [0x%llx, +%zu) overlaps chunk#%d at 0x%llx", (unsigned long long)va, c->size,
        //                  o.id, (unsigned long long)o.va);
        VMEM_CU(cuMemMap(va, c->size, 0, c->handle, 0));
        grant(va, c->size);
        c->va = va;
        log("map(chunk#%d, 0x%llx) -> [0x%llx, 0x%llx)", c->id, (unsigned long long)va, (unsigned long long)va,
            (unsigned long long)(va + c->size));
        dump();
        return va;
    }

    // memory stays, returns old address
    CUdeviceptr unmap(ChunkToken t) {
        Chunk *c = &get(t, "unmap");
        VMEM_REQUIRE(c->mapped(), "unmap: chunk#%d is not mapped", c->id);
        CUdeviceptr va = c->va;
        VMEM_CU(cuMemUnmap(va, c->size));
        c->va = 0;
        log("unmap(chunk#%d) from 0x%llx", c->id, (unsigned long long)va);
        dump();
        return va;
    }

    // move contents to `to`, same address; no-op if already there
    CUdeviceptr remap(ChunkToken t, Loc to) {
        Chunk *c = &get(t, "remap");
        VMEM_REQUIRE(c->mapped(), "remap: chunk#%d is not mapped (map it first)", c->id);
        if (c->loc == to) {
            fprintf(stderr, "[vmem] remap(chunk#%d, %s): already on %s, nothing to do\n", c->id, loc_name(to),
                    loc_name(to));
            return c->va;
        }
        CUmemAllocationProp p = prop(to);
        CUmemGenericAllocationHandle nh;
        VMEM_CU(cuMemCreate(&nh, c->size, &p, 0));

        // copy into new backing via a scratch range
        // ! blocking for now; the copy could run async, with the swap done once it finishes
        CUdeviceptr scratch = 0;
        VMEM_CU(cuMemAddressReserve(&scratch, c->size, gran_, 0, 0));
        VMEM_CU(cuMemMap(scratch, c->size, 0, nh, 0));
        grant(scratch, c->size);
        VMEM_CU(cuMemcpyDtoDAsync(scratch, c->va, c->size, copy_stream_));
        VMEM_CU(cuStreamSynchronize(copy_stream_));

        // swap backing under the same address
        VMEM_CU(cuMemUnmap(c->va, c->size));
        VMEM_CU(cuMemRelease(c->handle));
        VMEM_CU(cuMemMap(c->va, c->size, 0, nh, 0));
        grant(c->va, c->size);
        VMEM_CU(cuMemUnmap(scratch, c->size));
        VMEM_CU(cuMemAddressFree(scratch, c->size));

        log("remap(chunk#%d) %s -> %s at 0x%llx", c->id, loc_name(c->loc), loc_name(to), (unsigned long long)c->va);
        c->handle = nh;
        c->loc = to;
        dump();
        return c->va;
    }

    // must be unmapped first
    size_t release(ChunkToken t) {
        Chunk *c = &get(t, "release");
        VMEM_REQUIRE(!c->mapped(), "release: chunk#%d is still mapped at 0x%llx (unmap it first)", c->id,
                     (unsigned long long)c->va);
        VMEM_CU(cuMemRelease(c->handle));
        size_t sz = c->size;
        int id = c->id;
        chunks_.erase(id);   // token now stale
        log("release(chunk#%d) -> %zu MiB", id, sz >> 20);
        dump();
        return sz;
    }

    // same size as reserve(), nothing mapped inside
    size_t free(CUdeviceptr va, size_t size) {
        auto it = reservations_.find(va);
        VMEM_REQUIRE(it != reservations_.end(), "free: 0x%llx is not the start of a reservation", (unsigned long long)va);
        size_t sz = round_up(size);
        VMEM_REQUIRE(it->second == sz, "free: reservation at 0x%llx is %zu bytes, not %zu", (unsigned long long)va,
                     it->second, sz);
        for (auto &[id, c] : chunks_)
            VMEM_REQUIRE(!(c.mapped() && c.va >= va && c.va < va + sz),
                         "free: chunk#%d is still mapped inside [0x%llx, +%zu)", c.id, (unsigned long long)va, sz);
        VMEM_CU(cuMemAddressFree(va, sz));
        reservations_.erase(it);
        log("free(0x%llx, %zu) -> %zu MiB", (unsigned long long)va, size, sz >> 20);
        dump();
        return sz;
    }

    // ---- debug ----

    void print_state() const {
        size_t fr = 0, tot = 0;
        cuMemGetInfo(&fr, &tot);
        fprintf(stderr, "[vmem] state: VRAM free %zu / %zu MiB, %zu reservation(s), %zu chunk(s)\n", fr >> 20,
                tot >> 20, reservations_.size(), chunks_.size());
        for (auto &[va, sz] : reservations_)
            fprintf(stderr, "[vmem]   reservation 0x%llx +%zu MiB\n", (unsigned long long)va, sz >> 20);
        for (auto &[id, c] : chunks_) {
            if (c.mapped())
                fprintf(stderr, "[vmem]   chunk#%d %zu MiB %s, mapped at 0x%llx\n", id, c.size >> 20,
                        loc_name(c.loc), (unsigned long long)c.va);
            else
                fprintf(stderr, "[vmem]   chunk#%d %zu MiB %s, unmapped\n", id, c.size >> 20, loc_name(c.loc));
        }
    }

private:
    CUmemAllocationProp prop(Loc loc) const {
        CUmemAllocationProp p = {};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type = loc == Loc::Device ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA;
        p.location.id = loc == Loc::Device ? opt_.device : numa_;
        return p;
    }

    // device r/w, host chunks too (read over PCIe)
    void grant(CUdeviceptr va, size_t sz) const {
        CUmemAccessDesc a = {};
        a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        a.location.id = opt_.device;
        a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        VMEM_CU(cuMemSetAccess(va, sz, &a, 1));
    }

    size_t round_up(size_t x) const { return (x + gran_ - 1) / gran_ * gran_; }

    bool inside_reservation(CUdeviceptr va, size_t sz) const {
        // ! inefficient.. can maybe store as VA intervals?
        auto it = reservations_.upper_bound(va);
        if (it == reservations_.begin()) return false;
        --it;
        return va >= it->first && va + sz <= it->first + it->second;
    }

    // token -> chunk; rejects invalid, foreign, released
    Chunk &get(ChunkToken t, const char *what) {
        VMEM_REQUIRE(t.valid(), "%s: invalid chunk token", what);
        VMEM_REQUIRE(t.owner == serial_, "%s: chunk token belongs to another Manager", what);
        auto it = chunks_.find(t.id);
        VMEM_REQUIRE(it != chunks_.end(), "%s: chunk#%d was released", what, t.id);
        return it->second;
    }
    const Chunk &get(ChunkToken t, const char *what) const { return const_cast<Manager *>(this)->get(t, what); }

    static unsigned next_serial() { static unsigned n = 0; return ++n; }

    void log(const char *f, ...) const {
        if (!opt_.verbose) return;
        va_list ap; va_start(ap, f);
        fprintf(stderr, "[vmem] ");
        vfprintf(stderr, f, ap);
        fprintf(stderr, "\n");
        va_end(ap);
    }

    void dump() const { if (opt_.debug) print_state(); }

    // dtor helper: print a failed call, always (fprintf never throws)
    static bool ok(CUresult r, const char *call, int chunk) noexcept {
        if (r == CUDA_SUCCESS) return true;
        const char *n = "?";
        cuGetErrorName(r, &n);
        fprintf(stderr, "[vmem] shutdown: %s failed (chunk#%d): %s\n", call, chunk, n);
        return false;
    }

    Options opt_;
    unsigned serial_;
    CUdevice dev_ = 0;
    CUcontext ctx_ = nullptr;
    CUstream copy_stream_ = nullptr;
    int numa_ = 0;
    size_t gran_ = 0;
    int next_id_ = 0;
    std::map<int, Chunk> chunks_;
    std::map<CUdeviceptr, size_t> reservations_;   // base -> size
};

}  // namespace vmem
