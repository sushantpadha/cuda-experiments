// test 1: the basic workflow. reserve, create, map, fill, remap to host and back, check, tear down
// make test N=1
#include "pattern.cuh"
#include "vmem.cuh"

#include <cstdio>

// wrong words in the chunk
static unsigned long long count_bad(unsigned *p, size_t n) {
    unsigned long long bad = 0;
    pattern::check(p, n, 1, bad);
    return bad;
}

int main() {
    try {
        vmem::Manager m;
        const size_t size = 8 << 20, n = size / sizeof(unsigned);

        CUdeviceptr va = m.reserve(size);
        vmem::ChunkToken c = m.create(size, vmem::Loc::Device);
        m.map(c, va);

        pattern::fill(m.ptr<unsigned>(c), n, 1);
        int fails = 0;
        fails += count_bad(m.ptr<unsigned>(c), n) != 0;   // on device
        m.remap(c, vmem::Loc::Host);
        fails += count_bad(m.ptr<unsigned>(c), n) != 0;   // on host, same address
        m.remap(c, vmem::Loc::Device);
        fails += count_bad(m.ptr<unsigned>(c), n) != 0;   // back on device

        m.unmap(c);
        m.release(c);
        m.free(va);
        fprintf(stderr, fails ? "FAIL: %d of 3 checks\n" : "PASS\n", fails);
        return fails ? 1 : 0;
    } catch (const vmem::Error &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 2;
    }
}
