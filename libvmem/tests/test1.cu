// test 1: cuda backend. SmallPool reuses chunks, LargePool gets one chunk each, frees release, OOM returns nullptr
// make test N=1
#include "libvmem.cuh"
#include "pattern.cuh"

#include <cstdio>
#include <vector>

static int fails = 0;

static void expect(bool ok, const char *what) {
    if (!ok) fprintf(stderr, "  fail: %s\n", what);
    fails += !ok;
}

static bool intact(void *p, size_t bytes, unsigned k) {
    unsigned long long bad = 0;
    return pattern::check((unsigned *)p, bytes / 4, k, bad) == cudaSuccess && bad == 0;
}

int main() {
    try {
        vmem::init({.backend = vmem::Kind::Cuda});
        vmem::Allocator &a = vmem::get();

        // ---- SmallPool ----
        std::vector<std::pair<void *, size_t>> objs;
        for (unsigned i = 0; i < 64; i++) {
            size_t bytes = (size_t)(512 << (i % 12)) + 4 * i;   // 512 B .. 1 MiB, odd sizes
            if (bytes > (1 << 20)) bytes = 1 << 20;
            void *p = vmem::alloc(bytes);
            pattern::fill((unsigned *)p, bytes / 4, i + 1);
            objs.push_back({p, bytes});
        }
        size_t chunks = a.info().small_chunks;
        for (unsigned i = 0; i < 64; i += 2) vmem::free(objs[i].first);   // free half, then reuse slots
        for (unsigned i = 0; i < 64; i += 2) {
            objs[i].first = vmem::alloc(objs[i].second);
            pattern::fill((unsigned *)objs[i].first, objs[i].second / 4, i + 1);
        }
        expect(a.info().small_chunks <= chunks, "freed slots reused");
        bool all = true;
        for (unsigned i = 0; i < 64; i++) all &= intact(objs[i].first, objs[i].second, i + 1);
        expect(all, "small objects keep their data");
        vmem::evict(objs[0].first);   // hint on a small object: ignored
        expect(intact(objs[0].first, objs[0].second, 1), "small object intact after hint");

        void *dup = objs[1].first;
        vmem::free(dup);
        bool threw = false;
        try { vmem::free(dup); } catch (const vmem::Error &) { threw = true; }
        expect(threw, "double free caught");
        objs[1].first = nullptr;

        // ---- LargePool ----
        size_t big = 5 << 20;
        void *b = vmem::alloc(big);
        pattern::fill((unsigned *)b, big / 4, 7);
        expect(a.info().large_chunks == 1, "large allocation gets its own chunk");
        expect(a.info().usage.device >= 6 << 20, "large chunk rounded to 2 MiB");
        vmem::evict(b);
        vmem::prefetch(b);
        expect(intact(b, big, 7), "large allocation keeps its data");
        vmem::free(b);

        for (auto &[p, n] : objs) vmem::free(p);
        vmem::Info in = a.info();
        expect(in.small_chunks == 0 && in.large_chunks == 0 && in.usage.device == 0, "empty chunks released");

        // ---- VRAM full: cuda has no fallback ----
        std::vector<void *> fill;
        while (void *p = vmem::alloc(256 << 20)) fill.push_back(p);
        expect(!fill.empty(), "some VRAM before OOM");
        fprintf(stderr, "  %zu x 256 MiB before nullptr\n", fill.size());
        for (void *p : fill) vmem::free(p);
        expect(vmem::alloc(256 << 20) != nullptr, "VRAM usable again after OOM");

        vmem::shutdown();
        fprintf(stderr, fails ? "FAIL: %d check(s)\n" : "PASS\n", fails);
        return fails ? 1 : 0;
    } catch (const vmem::Error &e) {
        fprintf(stderr, "FAIL: %s (%s)\n", e.what(), vmem::status_name(e.status));
        return 2;
    }
}
