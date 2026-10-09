// runner data model: values, statements, the parsed program
#pragma once

#include "vmem.cuh"

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace runner {

[[noreturn]] inline void fail(const std::string &m) { throw std::runtime_error(m); }

// ---- values ----

enum class Kind { None, Addr, Chunk, Size, Loc, Bool };

inline const char *a_kind(Kind k) {
    static const char *n[] = {"nothing", "an address", "a chunk", "a size", "a location", "a bool"};
    return n[(int)k];
}

// what a variable holds at run time; kind None = no value (never set, or its line failed)
struct Value {
    Kind kind = Kind::None;
    uint64_t num = 0;       // address, size, bool, location (1 = host)
    vmem::ChunkToken tok;
    std::string text;       // shown instead of the value when set
};

inline std::string mib(uint64_t v) { return v % (1 << 20) ? std::format("{} B", v) : std::format("{} MiB", v >> 20); }

inline std::string show(const Value &v) {
    if (!v.text.empty()) return v.text;
    switch (v.kind) {
    case Kind::Addr: return std::format("{:#x}", v.num);
    case Kind::Chunk: return std::format("chunk#{}", v.tok.id);
    case Kind::Size: return mib(v.num);
    case Kind::Loc: return v.num ? "host" : "device";
    case Kind::Bool: return v.num ? "true" : "false";
    default: return "";
    }
}

// ---- program ----

// value = vars[var] (if var >= 0) + lit
struct Operand {
    int var = -1;
    uint64_t lit = 0;
};

struct Call;   // calls.cuh

struct Stmt {
    int line;
    std::string text;      // source, comment stripped
    const Call *call;
    int dst = -1;          // slot assigned, -1 = none
    Operand a, b;
    vmem::Loc loc = vmem::Loc::Device;
};

struct Program {
    std::vector<Stmt> stmts;
    std::vector<std::string> vars;     // slot -> name
    std::vector<std::string> errors;   // parse errors, one per bad line
};

}  // namespace runner
