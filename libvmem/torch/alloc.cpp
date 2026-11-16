// torch-vmem: libvmem backends as PyTorch's CUDA allocator (CUDAPluggableAllocator)
// - backend from VMEM_BACKEND=vmem|uvm|cuda (default vmem); vmem talks to vmemd at VMEM_DAEMON if one runs
// - large (> 1 MiB): one backend chunk per tensor, rounded to the granularity; can move (evict/prefetch)
// - small (<= 1 MiB): power-of-two slots carved from one granularity chunk; never move
// - free never returns memory: the block goes to a cache keyed by size
// v2 (events):
// - free records an event on the tensor's stream; reuse on another stream waits for it on the GPU, host never blocks
// - destroy and move wait for that block's event only, not for the whole device
// v3 (harness): vmem_touch records a last-use event per block after each op (torchvmem.Harness calls it)
// - evict/prefetch wait for that event, then move; with no touch since alloc they sync the device
// ponytail: record_stream is not seen (no C++ extension); a touch on a second stream covers it while the harness runs
// ponytail: small slabs are never destroyed; trim them too if small tensors ever pin real memory

#include "backends/cuda.h"
#include "backends/uvm.h"
#include "backends/vmem.h"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

using namespace vmem;

namespace {

constexpr size_t kSmallMax = 1 << 20, kMinObj = 512;

struct Block {
    Chunk c;                       // small: one slot (ptr, slot size); large: the backend chunk
    cudaStream_t stream = nullptr; // stream it was allocated on (PyTorch passes it to free)
    cudaEvent_t ev = nullptr;      // last free (cached) or last touch (live); created on first record
    bool recorded = false;         // ev covers the block's queued work
    bool touched = false;          // live and touched since alloc
    long long last_op = -1, uses = 0;   // harness state, reset at alloc
    bool small() const { return c.size <= kSmallMax; }
};

std::mutex mu;
std::unique_ptr<Backend> be;
std::map<uintptr_t, Block> live;                 // ordered, so any address inside a block finds it
std::map<size_t, std::vector<Block>> cache;      // free blocks by size
size_t live_bytes = 0, cached_bytes = 0;

[[noreturn]] void die(const char *what) {
    fprintf(stderr, "torch-vmem: %s\n", what);
    abort();
}

void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) die((std::string(what) + ": " + cudaGetErrorString(e)).c_str());
}

void record(Block &b, cudaStream_t s) {
    if (!b.ev) check(cudaEventCreateWithFlags(&b.ev, cudaEventDisableTiming), "cudaEventCreate");
    check(cudaEventRecord(b.ev, s), "cudaEventRecord");
    b.recorded = true;
}

// host waits until no queued kernel uses the block
void wait_idle(const Block &b, bool touched_only) {
    if (b.recorded && (b.touched || !touched_only)) check(cudaEventSynchronize(b.ev), "cudaEventSynchronize");
    else check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void init() {
    const char *k = getenv("VMEM_BACKEND");
    if (!k || !strcmp(k, "vmem")) be = std::make_unique<VmemBackend>();
    else if (!strcmp(k, "uvm")) be = std::make_unique<UvmBackend>();
    else if (!strcmp(k, "cuda")) be = std::make_unique<CudaBackend>();
    else die("VMEM_BACKEND must be vmem, uvm or cuda");
    const char *d = getenv("VMEM_DAEMON");
    if (be->open({0, d ? d : "/tmp/vmemd.sock", "torch-vmem", false}) != Status::Ok) die(be->last_error().c_str());
}

// destroy every cached large block; true if any went
bool trim() {
    bool any = false;
    for (auto it = cache.begin(); it != cache.end();) {
        if (it->first <= kSmallMax) { ++it; continue; }
        for (Block &b : it->second) {
            wait_idle(b, false);
            cached_bytes -= b.c.size;
            be->destroy(b.c);
            if (b.ev) cudaEventDestroy(b.ev);
            any = true;
        }
        it = cache.erase(it);
    }
    return any;
}

bool create(size_t size, Chunk &c) {
    Status s = be->create(size, c);
    if (s == Status::NoMemory && trim()) s = be->create(size, c);
    return s == Status::Ok;
}

void *alloc(size_t size, cudaStream_t stream) {
    if (!be) init();
    size_t g = be->granularity();
    bool small = size <= kSmallMax;
    size_t sz = small ? std::bit_ceil(std::max(size, kMinObj)) : (size + g - 1) / g * g;
    auto &list = cache[sz];
    if (list.empty()) {
        Chunk c;
        if (!create(small ? g : sz, c)) return nullptr;   // PyTorch raises its out-of-memory error
        for (size_t off = 0; off < c.size; off += sz) list.push_back({{(char *)c.ptr + off, sz, c.loc}, stream});
        cached_bytes += c.size;
    }
    // prefer a block last used on this stream: stream order makes it safe with no wait
    size_t i = list.size() - 1;
    for (size_t j = list.size(); j-- > 0;)
        if (list[j].stream == stream) { i = j; break; }   // ponytail: linear scan; index by stream if lists get long
    Block b = list[i];
    list.erase(list.begin() + i);
    if (b.stream != stream && b.recorded) check(cudaStreamWaitEvent(stream, b.ev, 0), "cudaStreamWaitEvent");
    b.stream = stream;
    b.touched = false;
    b.last_op = -1;
    b.uses = 0;
    live_bytes += sz;
    cached_bytes -= sz;
    live[(uintptr_t)b.c.ptr] = b;
    return b.c.ptr;
}

// the live block holding address p, or nullptr
Block *find(const void *p) {
    auto it = live.upper_bound((uintptr_t)p);
    if (it == live.begin()) return nullptr;
    --it;
    return (uintptr_t)p < it->first + it->second.c.size ? &it->second : nullptr;
}

int move(const void *p, Loc to) {
    Block *b = find(p);
    if (!b || b->small()) return (int)Status::Unsupported;
    if (b->c.loc == to) return (int)Status::Ok;
    wait_idle(*b, true);
    return (int)be->move(b->c, to);
}

}  // namespace

// ---- C ABI: PyTorch ----

extern "C" void *vmem_alloc(ssize_t size, int device, cudaStream_t stream) {
    if (device != 0) die("only device 0 is supported");
    if (size <= 0) return nullptr;
    std::lock_guard<std::mutex> g(mu);
    try {
        return alloc(size, stream);
    } catch (const std::exception &e) {
        die(e.what());
    }
}

extern "C" void vmem_free(void *p, ssize_t, int, cudaStream_t stream) {
    if (!p) return;
    std::lock_guard<std::mutex> g(mu);
    auto it = live.find((uintptr_t)p);
    if (it == live.end()) die("free of a pointer not from torch-vmem");
    Block b = it->second;
    live.erase(it);
    // a touch on another stream: make the free stream wait for it, so its event covers both
    if (b.touched && b.recorded) check(cudaStreamWaitEvent(stream, b.ev, 0), "cudaStreamWaitEvent");
    record(b, stream);
    b.stream = stream;
    live_bytes -= b.c.size;
    cached_bytes += b.c.size;
    cache[b.c.size].push_back(b);
}

// ---- C ABI: harness and hints (torchvmem.py) ----

// op `op` used these addresses; its kernels are queued on `stream`. unknown addresses are skipped
extern "C" void vmem_touch(const void *const *ptrs, int n, long long op, cudaStream_t stream) {
    std::lock_guard<std::mutex> g(mu);
    for (int i = 0; i < n; i++) {
        Block *b = find(ptrs[i]);
        if (!b || b->last_op == op) continue;   // views of one block in one op count once
        record(*b, stream);
        b->touched = true;
        b->last_op = op;
        b->uses++;
    }
}

// Status as int: 0 ok, 1 no memory (or vmemd refused), 2 unsupported (small, unknown, cuda backend), 4 fatal
extern "C" int vmem_evict(const void *p) {
    std::lock_guard<std::mutex> g(mu);
    return move(p, Loc::Host);
}

extern "C" int vmem_prefetch(const void *p) {
    std::lock_guard<std::mutex> g(mu);
    return move(p, Loc::Device);
}

// out: base, size, loc (0 device, 1 host), last_op, uses; false if p is not in a live block
extern "C" bool vmem_block(const void *p, long long out[5]) {
    std::lock_guard<std::mutex> g(mu);
    Block *b = find(p);
    if (!b) return false;
    out[0] = (long long)(uintptr_t)b->c.ptr;
    out[1] = b->c.size;
    out[2] = (int)b->c.loc;
    out[3] = b->last_op;
    out[4] = b->uses;
    return true;
}

// out: bytes held by live tensors, bytes cached, backend bytes on device, on host (-1 if the backend cannot tell)
extern "C" void vmem_info(long long out[4]) {
    std::lock_guard<std::mutex> g(mu);
    out[0] = live_bytes;
    out[1] = cached_bytes;
    out[2] = out[3] = -1;
    if (!be) return;
    try {
        out[2] = be->used(Loc::Device);
        out[3] = be->used(Loc::Host);
    } catch (const Error &) {}
}
