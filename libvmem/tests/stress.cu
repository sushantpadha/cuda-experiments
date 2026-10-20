// stress test. tiled matrix multiply, then a sharpen filter on big images (each VRAM/3 .. VRAM/2)
//   make stress ARGS="raw"      plain cudaMallocManaged, libvmem not used
//   make stress ARGS="cuda"     libvmem on a backend: cuda | uvm | vmemd
// flags: --resident   all images live at once, evict/prefetch around each use (default: one at a time)
//        --images N (4)   --n MATRIX_N (8192)   --data DIR (tests/data)
// prints time per phase, peak VRAM and DRAM, and a checksum that must match across modes
// the workload itself is in workload.cuh
#include "workload.cuh"

#include <sys/resource.h>
#include <unistd.h>

#include <atomic>
#include <thread>

// ---- peaks: device-wide VRAM used and system-wide DRAM used, sampled every 10 ms, minus the start ----

static size_t dram_used_kib() {
    FILE *f = fopen("/proc/meminfo", "r");
    size_t total = 0, avail = 0;
    char line[128];
    while (f && fgets(line, sizeof line, f)) {
        sscanf(line, "MemTotal: %zu", &total);
        sscanf(line, "MemAvailable: %zu", &avail);
    }
    if (f) fclose(f);
    return total - avail;
}

struct Peaks {
    size_t vram0 = 0, vram = 0, dram0 = 0, dram = 0;   // vram in bytes, dram in KiB
    std::atomic<bool> stop{false};
    std::thread th;

    void start() {
        size_t f = 0, t = 0;
        CHECK(cudaMemGetInfo(&f, &t));
        vram0 = vram = t - f;
        dram0 = dram = dram_used_kib();
        th = std::thread([this] {
            cudaSetDevice(0);
            while (!stop) {
                size_t f2 = 0, t2 = 0;
                if (cudaMemGetInfo(&f2, &t2) == cudaSuccess) vram = std::max(vram, t2 - f2);
                dram = std::max(dram, dram_used_kib());
                usleep(10000);
            }
        });
    }
    void end() { stop = true; th.join(); }
};

// ---- main ----

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string mode = argc > 1 ? argv[1] : "", data = "tests/data";
    bool resident = false;
    int images = 4, n = 8192;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--resident") resident = true;
        else if (a == "--images" && i + 1 < argc) images = atoi(argv[++i]);
        else if (a == "--n" && i + 1 < argc) n = atoi(argv[++i]);
        else if (a == "--data" && i + 1 < argc) data = argv[++i];
        else mode = "";
    }
    vmem::Kind kind = mode == "cuda" ? vmem::Kind::Cuda : mode == "uvm" ? vmem::Kind::Uvm : vmem::Kind::Vmem;
    if (mode != "raw" && mode != "cuda" && mode != "uvm" && mode != "vmemd") {
        fprintf(stderr, "usage: stress raw|cuda|uvm|vmemd [--resident] [--images N] [--n MATRIX_N] [--data DIR]\n");
        return 2;
    }
    if (n % TILE) { fprintf(stderr, "--n must be a multiple of %d\n", TILE); return 2; }
    raw = mode == "raw";
    std::vector<Base> bases = load_bases(data);
    if (bases.empty()) { fprintf(stderr, "no .pgm images in %s (run tests/data/make_images.py)\n", data.c_str()); return 2; }

    int rc = 0;
    Peaks pk;
    double t_mm = 0, t_img = 0;
    unsigned long long total = 0;
    try {
        unsetenv("VMEM_BACKEND");
        CHECK(cudaFree(0));
        if (!raw) vmem::init({.backend = kind, .name = "stress"});

        size_t f = 0, vram = 0;
        CHECK(cudaMemGetInfo(&f, &vram));
        const size_t W = 16384, H = vram * 5 / 12 / W;   // image = 5/12 of VRAM, between 1/3 and 1/2
        printf("mode: %s%s, VRAM %zu MiB, %d images of %zu MiB (%zux%zu), matrices %dx%d\n", mode.c_str(),
               resident ? " (resident)" : "", vram >> 20, images, (W * H) >> 20, W, H, n, n);

        pk.start();
        double t0 = now();
        printf("matrix multiply\n");
        rc |= matmul_phase(n);
        double t1 = now();
        printf("sharpen\n");
        rc |= sharpen_phase(resident, images, W, H, bases, total);
        double t2 = now();
        t_mm = t1 - t0;
        t_img = t2 - t1;
    } catch (const vmem::Error &e) {
        printf("FAIL: %s (%s)\n", e.what(), vmem::status_name(e.status));
        rc = 2;
    }
    if (pk.th.joinable()) pk.end();
    if (!raw) vmem::shutdown();

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("---- summary: %s%s\n", mode.c_str(), resident ? " --resident" : "");
    printf("time matmul   %.2f s\n", t_mm);
    printf("time sharpen  %.2f s\n", t_img);
    printf("time total    %.2f s\n", t_mm + t_img);
    printf("peak VRAM     %zu MiB   (device-wide used, minus start)\n", (pk.vram - pk.vram0) >> 20);
    printf("peak DRAM     %zu MiB   (system-wide used, minus start)\n", (pk.dram - pk.dram0) >> 10);
    printf("peak RSS      %ld MiB   (this process)\n", ru.ru_maxrss >> 10);
    printf("hints         %d ok, %d refused\n", hints_ok, hints_no);
    printf("checksum      %llx\n", total);
    printf(rc == 0 ? "PASS\n" : rc == 3 ? "OUT OF MEMORY\n" : "FAIL\n");
    return rc;
}
