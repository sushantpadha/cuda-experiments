// test 2: every call, data checked across remaps, error paths
// make test N=2, or make test N=2 ARGS=-d for state dumps
#include "pattern.cuh"
#include "vmem.cuh"

#include <cstdio>
#include <cstdlib>

static int failures = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { fprintf(stderr, "\tFAIL  %s\n", msg); ++failures; } \
                               else fprintf(stderr, "\tOK    %s\n", msg); } while (0)
#define EXPECT_THROW(stmt, msg) do { bool t_ = false; try { stmt; } catch (const vmem::Error &e) { \
        t_ = true; fprintf(stderr, "\t      (threw: %s)\n", e.what()); } EXPECT(t_, msg); } while (0)

// wrong words in the chunk at va, filled with key k
static unsigned long long count_bad(CUdeviceptr va, size_t bytes, unsigned k) {
    unsigned long long bad = ~0ull;
    pattern::check((const unsigned *)va, bytes / 4, k, bad);
    return bad;
}

int main(int argc, char **argv) {
    vmem::Options o;
    o.debug = argc > 1 && std::string(argv[1]) == "-d";
    try {
        vmem::Manager m(o);
        const size_t MiB = 1 << 20, G = m.granularity();

        CUdeviceptr va = m.reserve(64 * MiB);
        vmem::ChunkToken a = m.create(32 * MiB, vmem::Loc::Device);
        vmem::ChunkToken b = m.create(5 * MiB, vmem::Loc::Host);    // -> 6 MiB
        size_t asz = m.info(a).size, bsz = m.info(b).size;
        EXPECT(bsz % G == 0 && bsz >= 5 * MiB, "create rounds size up to granularity");
        EXPECT(a != b && !m.info(a).mapped(), "tokens are distinct; new chunk is unmapped");

        EXPECT(m.map(a, va) == va, "map returns the chunk's start address");
        CUdeviceptr next = va + asz;
        EXPECT(m.map(b, next) == next, "second chunk maps right after the first");
        EXPECT(m.on_device(a) && m.loc(b) == vmem::Loc::Host, "loc()/on_device() report residency");
        EXPECT(m.va(a) == va && m.ptr<unsigned>(b) == (unsigned *)next, "va() and ptr<T>() give the mapped address");

        pattern::fill(m.ptr<unsigned>(a), asz / 4, 1);
        pattern::fill(m.ptr<unsigned>(b), bsz / 4, 2);   // writes to host memory
        EXPECT(count_bad(m.va(a), asz, 1) == 0, "device chunk holds kernel-written data");
        EXPECT(count_bad(m.va(b), bsz, 2) == 0, "host chunk holds kernel-written data");

        EXPECT(m.remap(a, vmem::Loc::Host) == va, "remap returns the chunk's address");
        EXPECT(m.loc(a) == vmem::Loc::Host && m.va(a) == va, "remap device->host keeps the address and token");
        EXPECT(count_bad(m.va(a), asz, 1) == 0, "data survives device->host remap");
        EXPECT(m.remap(a, vmem::Loc::Host) == va && m.loc(a) == vmem::Loc::Host, "remap to current location is a no-op");
        m.remap(a, vmem::Loc::Device);
        EXPECT(m.on_device(a) && count_bad(m.va(a), asz, 1) == 0, "data survives host->device remap");
        m.remap(b, vmem::Loc::Device);
        EXPECT(m.on_device(b) && count_bad(m.va(b), bsz, 2) == 0, "host chunk remaps to device");

        EXPECT_THROW(m.map(a, va), "mapping an already-mapped chunk throws");
        EXPECT_THROW(m.release(a), "releasing a mapped chunk throws");
        EXPECT_THROW(m.free(va), "freeing a reservation with mapped chunks throws");
        EXPECT_THROW(m.map(m.create(2 * MiB, vmem::Loc::Device), va + 64 * MiB), "mapping outside a reservation throws");
        EXPECT_THROW(m.info(vmem::ChunkToken{}), "a default (invalid) token throws");
        {
            vmem::Options q = o; q.verbose = false;
            vmem::Manager other(q);
            vmem::ChunkToken foreign = other.create(2 * MiB, vmem::Loc::Device);
            EXPECT_THROW(m.info(foreign), "a token from another Manager throws");
        }

        EXPECT(m.unmap(a) == va, "unmap returns the address it was mapped at");
        EXPECT(!m.info(a).mapped(), "unmap clears the stored address");
        EXPECT_THROW(m.va(a), "va() of an unmapped chunk throws");
        EXPECT_THROW(m.remap(a, vmem::Loc::Host), "remapping an unmapped chunk throws");
        m.map(a, va);   // contents kept while unmapped
        EXPECT(count_bad(m.va(a), asz, 1) == 0, "data survives unmap + map");

        m.unmap(a);
        m.unmap(b);
        EXPECT(m.release(a) == 32 * MiB, "release returns the chunk size");
        vmem::ChunkToken c = m.create(32 * MiB, vmem::Loc::Device);   // new id, maybe a's memory
        EXPECT_THROW(m.unmap(a), "a released chunk's token throws, even after a new create");
        EXPECT(c != a, "new chunk gets a new token");
        m.release(c);
        m.release(b);
        EXPECT(m.free(va) == 64 * MiB, "free returns the reservation size");
        m.print_state();   // 1 chunk left, dtor releases it
    } catch (const vmem::Error &e) {
        fprintf(stderr, "UNEXPECTED vmem::Error: %s\n", e.what());
        return 2;
    }
    fprintf(stderr, failures ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
