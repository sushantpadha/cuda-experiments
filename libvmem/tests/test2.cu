// test 2: bookkeeping and error paths. calls that fail must say so and not crash
// the last step faults a kernel on purpose
//   bin/test2 c|u|v|d   (make test N=2c): cuda, uvm, vmem without a daemon, vmem with a vmemd it starts
#include "utils.cuh"

#include <cstring>
#include <vector>

__global__ void poke(int *p) { *p = 1; }

int main(int argc, char **argv) {
    return t::run("test2", argc > 1 ? argv[1] : nullptr, [] {
        vmem::Allocator &a = vmem::get();
        using vmem::Loc;
        using vmem::Status;
        bool uvm = !strcmp(a.info().backend, "uvm");
        bool moves = strcmp(a.info().backend, "cuda") != 0;
        bool daemon = a.info().link == vmem::Link::Connected;

        printf("usage: used() counts live chunks, rounded to 2 MiB\n");
        t::expect(a.used() == 0, "empty allocator holds 0 bytes");
        void *l = vmem::alloc(5 << 20);
        t::expect(a.used() == 6 << 20 && a.info().large_chunks == 1, "5 MiB large allocation = one 6 MiB chunk");
        void *s = vmem::alloc(100);
        t::expect(a.used() == 8 << 20 && a.info().small_chunks == 1, "100 B small object = one more 2 MiB chunk");
        if (uvm)
            t::expect_throws(Status::Unsupported, [&] { a.used(Loc::Device); }, "used(Loc) throws Unsupported on uvm");
        else
            t::expect(a.used(Loc::Device) + a.used(Loc::Host) == a.used(), "device + host = total");
        vmem::free(l);
        vmem::free(s);
        t::expect(a.used() == 0, "all chunks released after free");

        printf("errors: bad calls throw or return a status, none crash\n");
        vmem::free(nullptr);
        t::expect(vmem::alloc(0) == nullptr, "alloc(0) returns nullptr");
        void *s1 = vmem::alloc(100), *s2 = vmem::alloc(100);
        t::expect_throws(Status::Unsupported, [&] { vmem::free((char *)s1 + 1); }, "free inside an object throws");
        vmem::free(s1);
        t::expect_throws(Status::Unsupported, [&] { vmem::free(s1); }, "double free throws");
        int stack = 0;
        t::expect_throws(Status::Unsupported, [&] { vmem::free(&stack); }, "free of a foreign pointer throws");
        l = vmem::alloc(5 << 20);
        t::expect_throws(Status::Unsupported, [&] { vmem::free((char *)l + 4096); }, "free inside a large allocation throws");
        t::expect_throws(Status::Unsupported, [] { vmem::Allocator({.backend = vmem::Kind::Cuda, .small_max = 4 << 20}); },
                         "small_max larger than small_chunk throws");
        setenv("VMEM_BACKEND", "bogus", 1);
        t::expect_throws(Status::Unsupported, [] { vmem::Allocator(); }, "VMEM_BACKEND=bogus throws");
        unsetenv("VMEM_BACKEND");

        printf("hints: evict and prefetch return a status and never throw\n");
        t::expect(vmem::evict(&stack) == Status::Unsupported, "foreign pointer: Unsupported");
        t::expect(vmem::evict(s2) == Status::Unsupported, "small object: Unsupported");
        vmem::free(s2);
        t::put(l, 5 << 20, 7);
        Status e = vmem::evict(l);
        t::expect(e == (moves ? Status::Ok : Status::Unsupported), moves ? "large: evict Ok" : "large: evict Unsupported (cannot move)");
        t::expect(a.where(l) == (moves ? Loc::Host : Loc::Device), "location matches the result");
        t::expect(t::intact(l, 5 << 20, 7), "data intact after evict");
        t::expect(vmem::evict(l) == e && a.where(l) == (moves ? Loc::Host : Loc::Device), "second evict: same result, no move");
        Status p = vmem::prefetch(l);
        t::expect(p == (moves ? Status::Ok : Status::Unsupported), moves ? "large: prefetch Ok" : "large: prefetch Unsupported");
        t::expect(a.where(l) == Loc::Device && t::intact(l, 5 << 20, 7), "back on device, data intact");
        vmem::free(l);

        printf("out of memory: alloc returns nullptr, the allocator stays usable\n");
        t::expect(vmem::alloc(1ull << 50) == nullptr, "1 PiB request returns nullptr");
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        const size_t big = 256 << 20, cap = tot / big + 4;   // VRAM + 1 GiB: past the device, small enough to pin on host
        std::vector<void *> v;
        while (v.size() < cap)
            if (void *q = vmem::alloc(big)) v.push_back(q); else break;
        bool oom = v.size() < cap;
        printf("  %zu x 256 MiB allocated before %s\n", v.size(), oom ? "nullptr" : "the VRAM + 1 GiB cap, no failure");
        if (!moves) t::expect(oom, "cuda: runs out of VRAM");
        else if (uvm) t::expect(!oom, "uvm: no failure, the driver oversubscribes");
        else if (!daemon) t::expect(!oom && a.used(Loc::Host) > 0, "vmem local: no failure, the chunks past VRAM went to host");
        // with vmemd the broker's limits (5000 MiB device, 512 MiB host) must stop it, not the machine
        else {
            printf("  vmemd: %zu MiB device, %zu MiB host at the refusal\n", a.used(Loc::Device) >> 20, a.used(Loc::Host) >> 20);
            t::expect(oom, "vmem + vmemd: refused once the broker's limits are used up");
        }
        for (void *q : v) vmem::free(q);
        void *again = vmem::alloc(big);
        t::expect(again != nullptr, "allocation works again after OOM");
        vmem::free(again);

        printf("fault: a kernel writes to freed memory. libvmem must report Fatal at its next sync\n");
        void *live = vmem::alloc(3 << 20), *dead = vmem::alloc(3 << 20);   // large, so free(live) reaches a sync
        vmem::free(dead);
        printf("  if this crashes then we are good\n");
        fflush(stdout);
        poke<<<1, 1>>>((int *)dead);
        t::expect_throws(Status::Fatal, [&] { vmem::free(live); }, "free after the fault throws Fatal");
        t::expect_throws(Status::Fatal, [] { vmem::alloc(16); }, "allocator is dead, alloc throws Fatal");
    });
}
