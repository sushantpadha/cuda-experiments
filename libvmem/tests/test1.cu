// test 1: simple correctness, every allocation keeps its own data
//   bin/test1 c|u|v|d   (make test N=1c): cuda, uvm, vmem without a daemon, vmem with a vmemd it starts
#include "utils.cuh"

#include <vector>

int main(int argc, char **argv) {
    return t::run("test1", argc > 1 ? argv[1] : nullptr, [] {
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
        t::expect(a.info().small_chunks == 0, "every chunk went back once its last object was freed");

        printf("SmallPool: 100 objects of 64 KiB, 32 per 2 MiB chunk\n");
        o.clear();
        for (unsigned i = 0; i < 100; i++) {
            o.push_back({vmem::alloc(64 << 10), 64 << 10, 200 + i});
            t::put(o[i].p, o[i].n, o[i].k);
        }
        t::expect(a.info().small_chunks == 4, "one size class spans 4 chunks");
        t::expect(all_intact(), "objects in different chunks of one class do not overlap");
        for (Obj &x : o) vmem::free(x.p);
        t::expect(a.info().small_chunks == 0 && a.used() == 0, "all 4 chunks went back");

        printf("pool boundary: 1 MiB is small, 1 MiB + 1 B is large\n");
        void *s = vmem::alloc(1 << 20), *l = vmem::alloc((1 << 20) + 1);
        t::expect(a.info().small_chunks == 1 && a.info().large_chunks == 1 && a.used() == 4 << 20,
                  "one 2 MiB slab chunk and one 2 MiB large chunk");
        vmem::free(s);
        vmem::free(l);

        printf("LargePool: 5 MiB and 40 MiB, each with its own pattern\n");
        void *b1 = vmem::alloc(5 << 20), *b2 = vmem::alloc(40 << 20);
        t::put(b1, 5 << 20, 1);
        t::put(b2, 40 << 20, 2);
        t::expect(t::intact(b1, 5 << 20, 1) && t::intact(b2, 40 << 20, 2), "allocations are separate");
        vmem::free(b1);
        vmem::free(b2);
    });
}
