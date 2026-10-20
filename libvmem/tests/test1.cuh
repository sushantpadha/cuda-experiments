// test 1 body: simple correctness, every allocation keeps its own data
// run by test1c (cuda), test1u (uvm), test1v (vmem): make test N=1c
#pragma once

#include "utils.cuh"

#include <vector>

inline int test1(const char *name, vmem::Kind kind) {
    return t::run(name, kind, [] {
        vmem::Allocator &a = vmem::get();
        struct Obj { void *p; size_t n; unsigned k; };

        printf("SmallPool: 32 objects of 512 B .. 1 MiB, each filled with its own pattern\n");
        std::vector<Obj> o;
        for (unsigned i = 0; i < 32; i++) {
            size_t n = (512u << (i % 12)) - 4 * (i % 3);
            o.push_back({vmem::alloc(n), n, i + 1});
            t::put(o[i].p, n, o[i].k);
        }
        auto all_intact = [&] {
            bool ok = true;
            for (Obj &x : o) ok &= t::intact(x.p, x.n, x.k);
            return ok;
        };
        t::expect(all_intact(), "no object overlaps another");

        printf("SmallPool: free every other object, allocate the same sizes again\n");
        size_t chunks = a.info().small_chunks;
        for (unsigned i = 0; i < 32; i += 2) vmem::free(o[i].p);
        for (unsigned i = 0; i < 32; i += 2) {
            o[i].p = vmem::alloc(o[i].n);
            o[i].k += 100;
            t::put(o[i].p, o[i].n, o[i].k);
        }
        t::expect(all_intact(), "reused slots do not clobber live objects");
        t::expect(a.info().small_chunks == chunks, "freed slots were reused, no new chunk");
        for (Obj &x : o) vmem::free(x.p);

        printf("LargePool: 5 MiB and 40 MiB, each with its own pattern\n");
        void *b1 = vmem::alloc(5 << 20), *b2 = vmem::alloc(40 << 20);
        t::put(b1, 5 << 20, 1);
        t::put(b2, 40 << 20, 2);
        t::expect(t::intact(b1, 5 << 20, 1) && t::intact(b2, 40 << 20, 2), "allocations are separate");
        vmem::free(b1);
        vmem::free(b2);
    });
}
