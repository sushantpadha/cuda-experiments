// libvmem: per-process allocator over one backend (vmem | uvm | cuda)
// v1 impl (similar to pytorch):
// - SmallPool: <= small_max, power-of-two objects packed into small_chunk chunks, reused, never move
// - LargePool: > small_max, one chunk per allocation, destroyed on free unless freed with Reuse::Later
// ! all synchronization happens here: cuCtxSynchronize before every backend move and destroy
// single-threaded

// hints (evict, prefetch, free(p, Reuse::Later)) only mark chunks; libvmem decides when to move or cache
// TODO: testing
// TODO: vmem, uvm backends
// TODO: MPS support?

#pragma once

#include "backend.h"
#include "backends/cuda.h"

#include <cuda.h>

#include <bit>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace vmem {

enum class Kind { Vmem, Uvm, Cuda };

// free(): will the app want a block like this again?
enum class Reuse { No, Later };

// what the app told us about a chunk; libvmem moves Cold ones to host under pressure
enum class Temp { Hot, Cold, Cached };   // Cached = freed with Reuse::Later, data dead

struct Config {
    Kind backend = Kind::Vmem;              // VMEM_BACKEND=vmem|uvm|cuda overrides
    int device = 0;
    size_t small_max = 1 << 20;             // at most this goes to SmallPool
    size_t small_chunk = 2 << 20;
    size_t min_obj = 512;
    const char *daemon = "/tmp/vmemd.sock";
};

// pretty info
struct Info {
    const char *backend;
    Link link;
    Usage usage;
    size_t small_chunks, large_chunks;
};

class Allocator {
    // wraps one backend chunk (abstraction from the backend)
    // either contains obj-sized slots (SmallPool) where obj is the object size
    // or is a single chunk (LargePool) and obj==0
    struct Region {
        Chunk c;
        size_t obj;
        Temp temp = Temp::Hot;   // LargePool only
        std::vector<uint32_t> free_slots;
        std::vector<bool> busy;
        uint32_t used = 0;
    };

public:
    explicit Allocator(Config cfg = {}) : cfg_(cfg), b_(make_backend(env_kind(cfg.backend))) {
        // ! vmemd socket; ignored by backends without a daemon
        Status s = b_->open({cfg_.device, cfg_.daemon});
        if (s != Status::Ok) throw Error(s, std::string(b_->name()) + " open: " + b_->last_error());
        cfg_.small_chunk = round_up(cfg_.small_chunk);
        if (std::bit_ceil(cfg_.small_max) > cfg_.small_chunk)
            throw Error(Status::Unsupported, "config: small_max does not fit in small_chunk");
    }

    // best effort, never throws
    ~Allocator() {
        if (!dead_ && cuCtxSynchronize() == CUDA_SUCCESS)
            for (auto &[base, r] : regions_) b_->destroy(r.c);
        b_->close();
    }

    Allocator(const Allocator &) = delete;
    Allocator &operator=(const Allocator &) = delete;

    // ---- api ----

    // ! return nullptr when the backend has no memory left
    void *alloc(size_t bytes) {
        live();
        if (bytes == 0) return nullptr;
        // large pool condition
        if (bytes > cfg_.small_max) {
            Region *r = add(round_up(bytes), 0);
            return r ? r->c.ptr : nullptr;
        }
        
        // small pool
        size_t obj = std::bit_ceil(std::max(bytes, cfg_.min_obj));
        auto &part = partial_[obj];
        // 1. check for partial chunks; if not present create one
        Region *r = part.empty() ? add(cfg_.small_chunk, obj) : &regions_.at(*part.begin());
        if (!r) return nullptr;
        // 2. find a free slot
        uint32_t slot = r->free_slots.back();
        r->free_slots.pop_back();
        r->busy[slot] = true;
        r->used++;
        // 3. update partial map
        if (r->free_slots.empty()) part.erase(addr(r->c.ptr));
        return (char *)r->c.ptr + slot * obj;
    }

    // empty chunks go back to the backend, unless reuse == Later (LargePool): then the chunk stays as Cached
    void free(void *p, Reuse reuse = Reuse::No) {
        if (!p) return;
        live();
        auto [base, r] = find(p, "free");
        
        // small pool state updates
        if (r->obj) {
            size_t off = addr(p) - base;
            uint32_t slot = off / r->obj;
            // 1. sanity check
            if (off % r->obj || !r->busy[slot]) fail(Status::Unsupported, "free: not a live object");
            
            // 2. update state
            r->busy[slot] = false;
            r->free_slots.push_back(slot);
            if (--r->used) {
                // if chunk is in use, return
                partial_[r->obj].insert(base);
                return;
            }
            partial_[r->obj].erase(base);
        } else if (addr(p) != base) {
            fail(Status::Unsupported, "free: not the start of an allocation");
        }
        
        // TODO: reuse == Later: mark r->temp = Cached and return; alloc() of a fitting size takes a Cached chunk first
        (void)reuse;

        // ! FENCE!
        sync();

        // 1./3. call backend destroy and delete region itself
        Status s = b_->destroy(r->c);
        regions_.erase(base);
        if (s != Status::Ok) fail(s, "destroy");
    }

    // ---- hints ----
    // app promises p is idle and wants its data later; nothing moves now
    void evict(void *p) { hint(p, Temp::Cold); }
    // app wants p soon
    // TODO: if p is on host and the device has room, move it back now
    void prefetch(void *p) { hint(p, Temp::Hot); }

    Loc where(void *p) { return find(p, "where").second->c.loc; }

    Info info() const {
        size_t small = 0;
        for (auto &[base, r] : regions_)
            small += (r.obj != 0);
        return {b_->name(), b_->link(), b_->usage(), small, regions_.size() - small};
    }

    Status reconnect() { return b_->reconnect(); }

private:
    static uintptr_t addr(const void *p) { return (uintptr_t)p; }

    size_t round_up(size_t x) const {
        size_t g = b_->granularity();
        return (x + g - 1) / g * g;
    }

    // new chunk; slab chunks start with every slot free. nullptr on no memory
    Region *add(size_t size, size_t obj) {
        Chunk c;
        Status s = b_->create(size, c);
        if (s == Status::NoMemory && reclaim(size)) s = b_->create(size, c);
        if (s == Status::NoMemory) return nullptr;
        if (s != Status::Ok) fail(s, "create");
        Region r{c, obj, {}, {}};
        if (obj) {
            uint32_t n = size / obj;
            r.busy.assign(n, false);
            for (uint32_t i = n; i-- > 0;) r.free_slots.push_back(i);
            partial_[obj].insert(addr(c.ptr));
        }
        return &regions_.emplace(addr(c.ptr), std::move(r)).first->second; // return Region*
    }

    // small objects never move, so their hints are ignored
    void hint(void *p, Temp t) {
        live();
        Region *r = find(p, "hint").second;
        if (!r->obj) r->temp = t;
    }

    // device is full: free at least `need` bytes of VRAM; false if that is not possible
    // TODO: sync(), then destroy Cached chunks, then b_->move(c, Loc::Host) Cold ones (same address); skip if !b_->can_move()
    bool reclaim(size_t need) {
        (void)need;
        return false;
    }

    // a failed sync means the context is gone (sticky kernel error)
    void sync() {
        CUresult e = cuCtxSynchronize();
        if (e != CUDA_SUCCESS) {
            const char *s = "?";
            cuGetErrorString(e, &s);
            dead_ = true;
            throw Error(Status::Fatal, std::string("sync: ") + s);
        }
    }

    void live() const {
        if (dead_) throw Error(Status::Fatal, "allocator is dead after a fatal error; shut down");
    }

    [[noreturn]] void fail(Status s, const char *what) {
        if (s == Status::Fatal) dead_ = true;
        std::string msg = std::string(b_->name()) + " " + what;
        if (s != Status::Unsupported) msg += ": " + b_->last_error();
        throw Error(s, msg);
    }

    // address -> its region; bin searches in regions_
    std::pair<uintptr_t, Region *> find(void *p, const char *what) {
        auto it = regions_.upper_bound(addr(p));
        if (it == regions_.begin() || addr(p) >= std::prev(it)->first + std::prev(it)->second.c.size)
            fail(Status::Unsupported, (std::string(what) + ": pointer not from libvmem").c_str());
        --it;
        return {it->first, &it->second};
    }

    static Kind env_kind(Kind k) {
        const char *e = getenv("VMEM_BACKEND");
        if (!e) return k;
        if (!strcmp(e, "vmem")) return Kind::Vmem;
        if (!strcmp(e, "uvm")) return Kind::Uvm;
        if (!strcmp(e, "cuda")) return Kind::Cuda;
        throw Error(Status::Unsupported, std::string("VMEM_BACKEND=") + e + ": expected vmem, uvm or cuda");
    }

    // ! TODO write vmem and uvm backends
    static std::unique_ptr<Backend> make_backend(Kind k) {
        if (k == Kind::Cuda) return std::make_unique<CudaBackend>();
        throw Error(Status::Unsupported, k == Kind::Vmem ? "vmem backend not built yet" : "uvm backend not built yet");
    }

    Config cfg_;
    std::unique_ptr<Backend> b_;
    bool dead_ = false;
    // map for chunk base ptr -> region for both pools
    std::map<uintptr_t, Region> regions_;
    // map for object size -> chunks (base ptrs) with a free slot
    std::map<size_t, std::set<uintptr_t>> partial_;
};

// ---- ! singleton ----

inline std::unique_ptr<Allocator> &global_allocator() {
    static std::unique_ptr<Allocator> a;
    return a;
}

// ! runs on exit
inline void shutdown() { global_allocator().reset(); }

inline void init(Config cfg = {}) {
    static bool hooked = false;
    global_allocator() = std::make_unique<Allocator>(cfg);
    if (!hooked) hooked = !atexit(shutdown);
}

inline Allocator &get() {
    if (!global_allocator()) throw Error(Status::Unsupported, "libvmem: call vmem::init() first");
    return *global_allocator();
}

inline void *alloc(size_t bytes) { return get().alloc(bytes); }
template <class T> T *alloc(size_t n) { return static_cast<T *>(get().alloc(n * sizeof(T))); }
inline void free(void *p, Reuse reuse = Reuse::No) { get().free(p, reuse); }
inline void evict(void *p) { get().evict(p); }
inline void prefetch(void *p) { get().prefetch(p); }

}  // namespace vmem
