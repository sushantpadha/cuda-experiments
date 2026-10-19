// shared by the tests: check printing, pattern helpers, init/shutdown around a test body
#pragma once

#include "libvmem.cuh"
#include "pattern.cuh"

#include <cstdio>
#include <cstdlib>

namespace t {

inline int fails = 0;

inline void expect(bool ok, const char *what) {
    printf("  %s: %s\n", ok ? "ok" : "FAIL", what);
    fails += !ok;
}

// f must throw vmem::Error with this status
template <class F> void expect_throws(vmem::Status s, F f, const char *what) {
    bool ok = false;
    try { f(); } catch (const vmem::Error &e) { ok = e.status == s; }
    expect(ok, what);
}

inline void put(void *p, size_t bytes, unsigned k) { pattern::fill((unsigned *)p, bytes / 4, k); }

inline bool intact(void *p, size_t bytes, unsigned k) {
    unsigned long long bad = 1;
    return pattern::check((unsigned *)p, bytes / 4, k, bad) == cudaSuccess && bad == 0;
}

// the backend comes from the test, not from VMEM_BACKEND
template <class F> int run(vmem::Kind kind, F body) {
    unsetenv("VMEM_BACKEND");
    try {
        vmem::init({.backend = kind});
        printf("backend: %s\n", vmem::get().info().backend);
        body();
        vmem::shutdown();
    } catch (const vmem::Error &e) {
        printf("FAIL: uncaught %s (%s)\n", e.what(), vmem::status_name(e.status));
        return 2;
    }
    printf(fails ? "FAIL: %d check(s)\n" : "PASS\n", fails);
    return fails != 0;
}

}  // namespace t
