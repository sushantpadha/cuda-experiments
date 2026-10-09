// runner calls: what each .vm call does, as a table of name -> (argument shape, result kind, code)
#pragma once

#include "pattern.cuh"
#include "program.cuh"

#include <chrono>
#include <functional>
#include <map>
#include <thread>

namespace runner {

// ---- interpreter state ----

struct Ctx {
    vmem::Manager &m;
    const Program &p;
    std::vector<Value> vals;   // slot -> value

    const Value &get(int var, Kind k) const {
        if (vals[var].kind != k) fail(std::format("'{}' has no value (the line that set it failed)", p.vars[var]));
        return vals[var];
    }
    uint64_t size(const Operand &o) const { return (o.var >= 0 ? get(o.var, Kind::Size).num : 0) + o.lit; }
    CUdeviceptr addr(const Operand &o) const { return get(o.var, Kind::Addr).num + o.lit; }
    vmem::ChunkToken chunk(const Operand &o) const { return get(o.var, Kind::Chunk).tok; }
};

// ---- fill / check ----

inline void cuda(cudaError_t e) {
    if (e != cudaSuccess) fail(std::string("CUDA error: ") + cudaGetErrorName(e));
}

// pattern key is chunk id + 1, so chunk #0 is not all zeros
inline Value pattern_call(Ctx &c, const Stmt &s, bool fill) {
    vmem::ChunkToken t = c.chunk(s.a);
    vmem::ChunkInfo i = c.m.info(t);
    unsigned *p = c.m.ptr<unsigned>(t);
    size_t n = i.size / sizeof(unsigned);
    unsigned k = (unsigned)i.id + 1;
    if (fill) {
        cuda(pattern::fill(p, n, k));
        return {};
    }
    unsigned long long bad;
    cuda(pattern::check(p, n, k, bad));
    if (bad) fail(std::format("check: {} of {} words wrong in chunk#{}", bad, n, i.id));
    return {.text = "0 wrong"};
}

// ---- call table ----

// args, one letter each: S size, T time, L location, C chunk, A address (+ offset in map only)
struct Call {
    const char *args;
    Kind result;
    std::function<Value(Ctx &, const Stmt &)> run;
};

using V = Value;
inline const std::map<std::string, Call, std::less<>> CALLS = {
    {"reserve", {"S", Kind::Addr, [](Ctx &c, const Stmt &s) { return V{.num = c.m.reserve(c.size(s.a))}; }}},
    {"create", {"SL", Kind::Chunk, [](Ctx &c, const Stmt &s) { return V{.tok = c.m.create(c.size(s.a), s.loc)}; }}},
    {"map", {"CA", Kind::Addr, [](Ctx &c, const Stmt &s) { return V{.num = c.m.map(c.chunk(s.a), c.addr(s.b))}; }}},
    {"unmap", {"C", Kind::Addr, [](Ctx &c, const Stmt &s) { return V{.num = c.m.unmap(c.chunk(s.a))}; }}},
    {"remap", {"CL", Kind::Addr, [](Ctx &c, const Stmt &s) { return V{.num = c.m.remap(c.chunk(s.a), s.loc)}; }}},
    {"release", {"C", Kind::Size, [](Ctx &c, const Stmt &s) { return V{.num = c.m.release(c.chunk(s.a))}; }}},
    {"free", {"A", Kind::Size, [](Ctx &c, const Stmt &s) { return V{.num = c.m.free(c.addr(s.a))}; }}},
    {"va", {"C", Kind::Addr, [](Ctx &c, const Stmt &s) { return V{.num = c.m.va(c.chunk(s.a))}; }}},
    {"loc", {"C", Kind::Loc, [](Ctx &c, const Stmt &s) { return V{.num = !c.m.on_device(c.chunk(s.a))}; }}},
    {"on_device", {"C", Kind::Bool, [](Ctx &c, const Stmt &s) { return V{.num = c.m.on_device(c.chunk(s.a))}; }}},
    {"granularity", {"", Kind::Size, [](Ctx &c, const Stmt &) { return V{.num = c.m.granularity()}; }}},
    {"print_state", {"", Kind::None, [](Ctx &c, const Stmt &) { c.m.print_state(); return V{}; }}},
    {"fill", {"C", Kind::None, [](Ctx &c, const Stmt &s) { return pattern_call(c, s, true); }}},
    {"check", {"C", Kind::None, [](Ctx &c, const Stmt &s) { return pattern_call(c, s, false); }}},
    {"sleep", {"T", Kind::None, [](Ctx &, const Stmt &s) {
         std::this_thread::sleep_for(std::chrono::microseconds(s.a.lit));
         return V{};
     }}},
    {"info", {"C", Kind::None, [](Ctx &c, const Stmt &s) {
         vmem::ChunkInfo i = c.m.info(c.chunk(s.a));
         return V{.text = std::format("chunk#{} {} {}{}", i.id, mib(i.size), vmem::loc_name(i.loc),
                                      i.mapped() ? std::format(" at {:#x}", i.va) : " unmapped")};
     }}},
};

}  // namespace runner
