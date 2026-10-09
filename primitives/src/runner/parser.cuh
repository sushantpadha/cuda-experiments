// runner parser: one pass over a .vm file; checks syntax, call names, argument shapes, names and kinds
#pragma once

#include "calls.cuh"

#include <algorithm>
#include <istream>
#include <map>
#include <regex>
#include <sstream>

namespace runner {

inline std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// ---- one argument ----

// as written: name, number[unit], or name + number[unit]
struct Arg {
    std::string name, unit;
    bool plus = false, has_num = false;
    uint64_t num = 0;
};

inline Arg read_arg(const std::string &where, const std::string &a) {
    static const std::regex re(R"(^([A-Za-z_]\w*)?(\s*\+\s*)?(?:(\d+)([A-Za-z]*))?$)");
    std::smatch m;
    if (a.empty() || !std::regex_match(a, m, re) || (m[2].matched && !(m[1].matched && m[3].matched)))
        fail(std::format("{}: cannot read '{}'", where, a));
    Arg r{m[1], m[4], m[2].matched, m[3].matched};
    if (r.has_num) {
        if (m[3].length() > 15) fail("number too large");
        r.num = std::stoull(m[3]);
    }
    return r;
}

// number times its unit; units maps suffix -> multiplier
inline uint64_t scaled(const Arg &r, const std::map<std::string, uint64_t> &units, const std::string &where) {
    auto u = units.find(r.unit);
    if (u == units.end()) {
        std::string names;
        for (auto &[k, v] : units) if (!k.empty()) names += (names.empty() ? "" : ", ") + k;
        fail(std::format("{}: unknown unit '{}' (use {})", where, r.unit, names));
    }
    if (r.num > UINT64_MAX / u->second) fail(where + ": too large");
    return r.num * u->second;
}

// ---- statements ----

class Parser {
public:
    explicit Parser(Program &p) : p_(p) {}

    // skips blanks and comments; on error records it and goes on
    void line(int no, const std::string &raw) {
        std::string s = trim(raw.substr(0, raw.find('#')));
        if (s.empty()) return;
        try {
            stmt(no, s);
        } catch (const std::runtime_error &e) {
            p_.errors.push_back(std::format("line {}: {}\n    {}", no, e.what(), s));
        }
    }

private:
    // [name =] call(arg, ...)
    void stmt(int no, const std::string &s) {
        static const std::regex call_re(R"(^(?:([A-Za-z_]\w*)\s*=\s*)?([A-Za-z_]\w*)\s*\((.*)\)$)");
        std::smatch m;
        if (!std::regex_match(s, m, call_re)) fail("expected call(args) or name = call(args)");
        std::string dst = m[1], name = m[2], argstr = trim(m[3]);
        auto it = CALLS.find(name);
        if (it == CALLS.end()) fail(std::format("unknown call '{}'", name));
        const Call &c = it->second;

        std::vector<std::string> args;
        std::stringstream ss(argstr + ",");   // trailing comma so a final empty argument still counts
        for (std::string a; !argstr.empty() && std::getline(ss, a, ',');) args.push_back(trim(a));
        std::string shape = c.args;
        if (args.size() != shape.size())
            fail(std::format("{}() takes {} argument(s), got {}", name, shape.size(), args.size()));

        Stmt st{no, s, &c};
        for (size_t i = 0; i < shape.size(); ++i) {
            std::string where = std::format("{}() argument {}", name, i + 1);
            convert(where, shape[i], name == "map", read_arg(where, args[i]), i ? st.b : st.a, st.loc);
        }
        if (!dst.empty()) {
            if (c.result == Kind::None) fail(name + "() returns nothing to assign");
            if (dst == "device" || dst == "host" || CALLS.contains(dst)) fail(std::format("'{}' is a reserved word", dst));
            st.dst = slot(dst);
            kinds_[dst] = c.result;   // after the arguments, so x = f(x) sees the old x
        }
        p_.stmts.push_back(st);
    }

    // check one argument against its shape letter and fill the operand
    void convert(const std::string &where, char shape, bool offset_ok, const Arg &r, Operand &o, vmem::Loc &loc) {
        static const std::map<std::string, uint64_t> sizes = {{"", 1}, {"K", 1 << 10}, {"M", 1 << 20}, {"G", 1 << 30}},
                                                     times = {{"us", 1}, {"ms", 1000}, {"s", 1000000}};
        bool bare_name = !r.name.empty() && !r.has_num, bare_num = r.name.empty() && r.has_num;
        switch (shape) {
        case 'L':
            if (!bare_name || (r.name != "device" && r.name != "host")) fail(where + " must be device or host");
            loc = r.name == "host" ? vmem::Loc::Host : vmem::Loc::Device;
            return;
        case 'T':
            if (!bare_num) fail(where + " must be a time like 10ms");
            o.lit = scaled(r, times, where);
            return;
        case 'S':
            if (bare_num) o.lit = scaled(r, sizes, where);
            else if (bare_name) o.var = use(r.name, Kind::Size, where);
            else fail(where + " must be a size like 64M or a size variable");
            return;
        case 'C':
            if (!bare_name) fail(where + " must be a chunk variable");
            o.var = use(r.name, Kind::Chunk, where);
            return;
        case 'A':
            if (r.name.empty()) fail(where + " must be an address variable");
            if (r.plus && !offset_ok) fail(where + " takes no offset");
            o.var = use(r.name, Kind::Addr, where);
            if (r.plus) o.lit = scaled(r, sizes, where);
            return;
        }
    }

    // input variable: must exist here, with the right kind
    int use(const std::string &name, Kind want, const std::string &where) {
        auto it = kinds_.find(name);
        if (it == kinds_.end()) fail(std::format("'{}' is not defined", name));
        if (it->second != want)
            fail(std::format("{} must be {}, '{}' is {}", where, a_kind(want), name, a_kind(it->second)));
        return slot(name);
    }

    int slot(const std::string &name) {
        auto it = std::ranges::find(p_.vars, name);
        if (it != p_.vars.end()) return int(it - p_.vars.begin());
        p_.vars.push_back(name);
        return int(p_.vars.size()) - 1;
    }

    Program &p_;
    std::map<std::string, Kind> kinds_;   // kind of each name at this point in the file
};

inline Program parse(std::istream &in) {
    Program p;
    Parser ps(p);
    int no = 0;
    for (std::string l; std::getline(in, l);) ps.line(++no, l);
    return p;
}

}  // namespace runner
