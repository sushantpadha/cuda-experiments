// runner driver: parse a file, open the logs, run every statement on one Manager, report
#pragma once

#include "log.cuh"
#include "parser.cuh"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <memory>

namespace runner {

struct Options {
    bool verbose = false, debug = false, stable = false, pid = false, parse_only = false;
    std::string logdir = "logs", file;
};

// CUDA error name if the context is broken (sticky error), else ""
inline std::string broken_context() {
    CUresult r = cuCtxSynchronize();
    if (r == CUDA_SUCCESS) return "";
    const char *n = "?";
    cuGetErrorName(r, &n);
    return n;
}

struct Outcome {
    bool ok;
    std::string result;   // "-> value", or the error message
};

// one statement; a failed line leaves its variable without a value
inline Outcome execute(Ctx &ctx, const Stmt &s) {
    try {
        Value v = s.call->run(ctx, s);
        v.kind = s.call->result;
        if (s.dst >= 0) ctx.vals[s.dst] = v;
        std::string shown = show(v);
        return {true, shown.empty() ? "" : "-> " + shown};
    } catch (const std::exception &e) {
        if (s.dst >= 0) ctx.vals[s.dst] = {};
        return {false, e.what()};
    }
}

inline KeyValues run_header(const Options &o, const Program &p, const std::string &base) {
    std::string opts;
    for (auto [on, flag] : {std::pair{o.verbose, "-v"}, {o.debug, "-d"}, {o.stable, "--stable"}, {o.pid, "--pid"}})
        if (on) opts += opts.empty() ? flag : std::string(" ") + flag;
    KeyValues kv = {{"runner", o.file}};
    if (!o.stable) {
        char when[32];
        std::time_t now = std::time(nullptr);
        std::strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        kv.push_back({"started", when});
    }
    kv.push_back({"options", opts.empty() ? "none" : opts});
    kv.push_back({"program", std::format("{} statement(s), {} variable(s)", p.stmts.size(), p.vars.size())});
    kv.push_back({"log", base + ".log"});
    if (o.debug) kv.push_back({"debug", base + ".debug.log"});
    return kv;
}

// exit code: 0 all ok, 1 a statement or the Manager failed, 2 bad file or options
inline int run(const Options &o) {
    // ---- parse ----
    std::ifstream in(o.file);
    if (!in) { fprintf(stderr, "runner: cannot open %s\n", o.file.c_str()); return 2; }
    Program p = parse(in);
    for (auto &e : p.errors) fprintf(stderr, "%s: %s\n", o.file.c_str(), e.c_str());
    if (!p.errors.empty()) { fprintf(stderr, "%zu error(s), nothing run\n", p.errors.size()); return 2; }
    if (o.parse_only) { fprintf(stderr, "%s: %zu statement(s), ok\n", o.file.c_str(), p.stmts.size()); return 0; }

    // ---- logs ----
    std::error_code ec;
    std::filesystem::create_directories(o.logdir, ec);
    std::string base = o.logdir + "/" + std::filesystem::path(o.file).stem().string();
    if (o.pid) base += std::format(".{}", getpid());
    LogFile log(base + ".log", o.stable);
    std::unique_ptr<LogFile> dbg;
    if (o.debug) dbg = std::make_unique<LogFile>(base + ".debug.log", o.stable);
    if (!log.good() || (dbg && !dbg->good())) { fprintf(stderr, "runner: cannot write %s.log\n", base.c_str()); return 2; }
    // to the log (and stdout with -v); the debug log also gets the state
    auto emit = [&](const std::string &text, const std::string &state = "") {
        std::string shown = log.put(text);
        if (dbg) dbg->put(text + indent(state));
        if (o.verbose) std::cout << shown << std::flush;
    };
    auto say = [&](const std::string &text) { if (!o.verbose) fputs(text.c_str(), stderr); };   // terminal summary

    KeyValues head = run_header(o, p, base);
    emit(keyvalues(head, "# "));
    say(keyvalues(head, ""));

    // ---- manager; its output is collected per line ----
    std::string out;
    vmem::Options mo;
    mo.out = [&out](const std::string &l) { out += l + "\n"; };
    using clk = std::chrono::steady_clock;
    auto since = [](clk::time_point t) { return std::chrono::duration<double, std::micro>(clk::now() - t).count(); };
    auto time = [&](double us) { return o.stable ? std::string("-") : std::format("{:10.1f} us", us); };

    auto t0 = clk::now();
    std::unique_ptr<vmem::Manager> m;
    std::string err;
    try { m = std::make_unique<vmem::Manager>(mo); } catch (const std::exception &e) { err = e.what(); }
    emit(block("init", "Manager()", m ? "ok" : "ERROR", time(since(t0)), err, out));
    if (!m) { say(std::format("result   FAIL: Manager() failed: {}\n", err)); return 1; }

    // ---- statements ----
    Ctx ctx{*m, p, std::vector<Value>(p.vars.size())};
    int errors = 0, skipped = 0;
    double total = 0;
    bool dead = false;   // context broken: every later call fails, so skip the rest
    for (const Stmt &s : p.stmts) {
        std::string tag = std::to_string(s.line);
        if (dead) {
            ++skipped;
            emit(block(tag, s.text, "SKIP", "-", "", ""));
            continue;
        }
        out.clear();
        t0 = clk::now();
        Outcome r = execute(ctx, s);
        double us = since(t0);
        if (!r.ok) {
            ++errors;
            if (std::string n = broken_context(); !n.empty()) {
                dead = true;
                r.result += std::format("; CUDA context is broken ({}), skipping the rest", n);
            }
        }
        total += us;
        std::string line_out = out, state;
        if (dbg) { out.clear(); m->print_state(); state = out; }
        emit(block(tag, s.text, r.ok ? "ok" : "ERROR", time(us), r.result, line_out), state);
    }

    // ---- shutdown ----
    out.clear();
    m.reset();
    bool clean = out.find(" failed (") == std::string::npos;   // the dtor prints each failed cleanup
    emit(block("end", "~Manager()", clean ? "ok" : "ERROR", time(0), "", out));

    std::string sum = std::format("{}: {} statement(s), {} error(s)", errors ? "FAIL" : "PASS", p.stmts.size(), errors);
    if (skipped) sum += std::format(", {} skipped", skipped);
    if (!o.stable) sum += std::format(", {:.1f} ms in statements", total / 1000);
    emit(keyvalues({{"result", sum}}, "# "));
    say(keyvalues({{"result", sum}}, ""));
    return errors ? 1 : 0;
}

}  // namespace runner
