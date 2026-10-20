// test 3: stress. tiled matrix multiply, then a sharpen filter on big images (each VRAM/3 .. VRAM/2)
//   make test N=3 ARGS="raw"      plain cudaMallocManaged, libvmem not used
//   make test N=3 ARGS="cuda"     libvmem on a backend: cuda | uvm | vmemd
// flags: --resident   all images live at once, evict/prefetch around each use (default: one at a time)
//        --images N (4)   --n MATRIX_N (8192)   --data DIR (tests/data)
// prints time per phase, peak VRAM and DRAM, and a checksum that must match across modes
#include "libvmem.cuh"

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#define CHECK(call)                                                                                  \
    do {                                                                                             \
        cudaError_t e_ = (call);                                                                     \
        if (e_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            exit(2);                                                                                 \
        }                                                                                            \
    } while (0)

// ---- memory: raw managed memory, or libvmem ----

static bool raw = false;
static int hints_ok = 0, hints_no = 0;

static void *mem_alloc(size_t n) {
    if (!raw) return vmem::alloc(n);
    void *p = nullptr;
    if (cudaMallocManaged(&p, n) == cudaSuccess) return p;
    cudaGetLastError();
    return nullptr;
}
static void mem_free(void *p) {
    if (raw) CHECK(cudaFree(p)); else vmem::free(p);
}
// raw has no hints
static void hint(vmem::Status s) { (s == vmem::Status::Ok ? hints_ok : hints_no)++; }
static void mem_evict(void *p) { if (!raw) hint(vmem::evict(p)); }
static void mem_prefetch(void *p) { if (!raw) hint(vmem::prefetch(p)); }

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

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- matrix multiply: C = A * B, tiled through shared memory ----

constexpr int TILE = 32;

// A[r][c] = (r + 2c) % 5, B[r][c] = (3r + c) % 4: small integers, so float sums are exact
__global__ void init_mat(float *m, int n, int which) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)n * n; i += (size_t)gridDim.x * blockDim.x) {
        size_t r = i / n, c = i % n;
        m[i] = which == 0 ? (r + 2 * c) % 5 : (3 * r + c) % 4;
    }
}

// n must be a multiple of TILE
__global__ void matmul(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0;
    for (int t = 0; t < n / TILE; t++) {
        As[ty][tx] = A[(size_t)row * n + t * TILE + tx];
        Bs[ty][tx] = B[(size_t)(t * TILE + ty) * n + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[(size_t)row * n + col] = acc;
}

// returns 0 ok, 1 wrong result, 3 out of memory
static int matmul_phase(int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    float *m[3];
    for (int i = 0; i < 3; i++) {
        m[i] = (float *)mem_alloc(bytes);
        if (!m[i]) {
            printf("  OOM: matrix %d of 3 (%zu MiB each)\n", i, bytes >> 20);
            for (int j = 0; j < i; j++) mem_free(m[j]);
            return 3;
        }
    }
    init_mat<<<1024, 256>>>(m[0], n, 0);
    init_mat<<<1024, 256>>>(m[1], n, 1);
    matmul<<<dim3(n / TILE, n / TILE), dim3(TILE, TILE)>>>(m[0], m[1], m[2], n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // spot check: corners, middle, and a few scattered entries against a host dot product
    int bad = 0, samples = 0;
    for (int s = 0; s < 12; s++) {
        size_t r = s == 0 ? 0 : s == 1 ? n - 1 : (size_t)s * 2654435761u % n;
        size_t c = s == 0 ? 0 : s == 1 ? n - 1 : (size_t)s * 40503u % n;
        long want = 0;
        for (size_t k = 0; k < (size_t)n; k++) want += (long)((r + 2 * k) % 5) * ((3 * k + c) % 4);
        float got = 0;
        CHECK(cudaMemcpy(&got, m[2] + r * n + c, sizeof got, cudaMemcpyDeviceToHost));
        bad += got != (float)want;
        samples++;
    }
    printf("  C = A*B, %dx%d, 3 x %zu MiB: %d of %d spot checks wrong\n", n, n, bytes >> 20, bad, samples);
    for (int i = 0; i < 3; i++) mem_free(m[i]);
    return bad != 0;
}

// ---- images ----

struct Base {
    std::string name;
    int w = 0, h = 0;
    std::vector<unsigned char> px;
};

static std::vector<Base> load_bases(const std::string &dir) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::exists(dir))
        for (auto &e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".pgm") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    std::vector<Base> v;
    for (auto &p : files) {
        FILE *f = fopen(p.c_str(), "rb");
        Base b;
        int maxv = 0;
        if (!f || fscanf(f, "P5 %d %d %d%*c", &b.w, &b.h, &maxv) != 3 || maxv != 255) {
            fprintf(stderr, "bad pgm: %s\n", p.c_str());
            exit(2);
        }
        b.name = p.stem().string();
        b.px.resize((size_t)b.w * b.h);
        if (fread(b.px.data(), 1, b.px.size(), f) != b.px.size()) exit(2);
        fclose(f);
        v.push_back(std::move(b));
    }
    return v;
}

// big image = the base image tiled over W x H, each tile shifted so tiles differ
__global__ void expand(const unsigned char *base, int bw, int bh, unsigned char *img, size_t W, size_t H) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < W * H; i += (size_t)gridDim.x * blockDim.x) {
        size_t y = i / W, x = i % W;
        img[i] = (unsigned char)(base[(y % bh) * bw + x % bw] + 31 * (x / bw) + 17 * (y / bh));
    }
}

// 3x3 sharpen: 5*center - 4 neighbours, edges clamped
__device__ __host__ inline int sharpen_px(int c, int l, int r, int u, int d) {
    int v = 5 * c - l - r - u - d;
    return v < 0 ? 0 : v > 255 ? 255 : v;
}

__global__ void sharpen(const unsigned char *in, unsigned char *out, size_t W, size_t H) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < W * H; i += (size_t)gridDim.x * blockDim.x) {
        size_t y = i / W, x = i % W;
        size_t xl = x ? x - 1 : 0, xr = x + 1 < W ? x + 1 : W - 1, yu = y ? y - 1 : 0, yd = y + 1 < H ? y + 1 : H - 1;
        out[i] = sharpen_px(in[i], in[y * W + xl], in[y * W + xr], in[yu * W + x], in[yd * W + x]);
    }
}

__device__ unsigned long long d_sum;

__global__ void sum_k(const unsigned char *p, size_t n) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) s += p[i];
    atomicAdd(&d_sum, s);
}

static unsigned long long checksum(const unsigned char *p, size_t n) {
    unsigned long long z = 0, s = 0;
    CHECK(cudaMemcpyToSymbol(d_sum, &z, sizeof z));
    sum_k<<<1024, 256>>>(p, n);
    CHECK(cudaMemcpyFromSymbol(&s, d_sum, sizeof s));
    return s;
}

// recompute some output pixels on the host from the input and compare; returns wrong count
static int verify(const unsigned char *in, const unsigned char *out, size_t W, size_t H) {
    auto at = [&](const unsigned char *d, size_t x, size_t y) {
        unsigned char v;
        CHECK(cudaMemcpy(&v, d + y * W + x, 1, cudaMemcpyDeviceToHost));
        return (int)v;
    };
    int bad = 0;
    for (int s = 0; s < 64; s++) {
        size_t x = s == 0 ? 0 : s == 1 ? W - 1 : (size_t)s * 2654435761u % W;
        size_t y = s == 0 ? 0 : s == 1 ? H - 1 : (size_t)s * 40503u % H;
        size_t xl = x ? x - 1 : 0, xr = x + 1 < W ? x + 1 : W - 1, yu = y ? y - 1 : 0, yd = y + 1 < H ? y + 1 : H - 1;
        int want = sharpen_px(at(in, x, y), at(in, xl, y), at(in, xr, y), at(in, x, yu), at(in, x, yd));
        bad += at(out, x, y) != want;
    }
    return bad;
}

// returns 0 ok, 1 wrong result, 3 out of memory
static int sharpen_phase(bool resident, int count, size_t W, size_t H, const std::vector<Base> &bases, unsigned long long &total) {
    const size_t X = W * H;
    std::vector<unsigned char *> in(count, nullptr);
    int rc = 0;

    auto generate = [&](int i) {
        const Base &b = bases[i % bases.size()];
        unsigned char *d = (unsigned char *)mem_alloc(b.px.size());
        if (!d) return false;
        CHECK(cudaMemcpy(d, b.px.data(), b.px.size(), cudaMemcpyHostToDevice));
        expand<<<1024, 256>>>(d, b.w, b.h, in[i], W, H);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        mem_free(d);
        return true;
    };
    auto oom = [&](const char *what, int i) {
        printf("  OOM: %s for image %d\n", what, i);
        rc = 3;
    };

    if (resident)   // every image lives at once; hint each one away after writing it
        for (int i = 0; i < count && !rc; i++) {
            if (!(in[i] = (unsigned char *)mem_alloc(X))) oom("input", i);
            else if (!generate(i)) oom("base image", i);
            else mem_evict(in[i]);
        }

    for (int i = 0; i < count && !rc; i++) {
        double t0 = now();
        if (!resident) {
            if (!(in[i] = (unsigned char *)mem_alloc(X))) { oom("input", i); break; }
            if (!generate(i)) { oom("base image", i); break; }
        } else {
            mem_prefetch(in[i]);
        }
        double t1 = now();
        unsigned char *out = (unsigned char *)mem_alloc(X);
        if (!out) { oom("output", i); break; }
        sharpen<<<1024, 256>>>(in[i], out, W, H);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        double t2 = now();
        unsigned long long s = checksum(out, X);
        int bad = verify(in[i], out, W, H);
        total += s;
        rc |= bad != 0;
        printf("  image %d: %-10s %zux%zu %zu MiB  prepare %.2fs  sharpen %.2fs  sum %llx  %d of 64 spot checks wrong\n", i,
               bases[i % bases.size()].name.c_str(), W, H, X >> 20, t1 - t0, t2 - t1, s, bad);
        mem_free(out);
        if (resident) mem_evict(in[i]); else mem_free(in[i]);
        in[i] = resident ? in[i] : nullptr;
    }
    for (unsigned char *p : in)
        if (p) mem_free(p);
    return rc;
}

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
        fprintf(stderr, "usage: test3 raw|cuda|uvm|vmemd [--resident] [--images N] [--n MATRIX_N] [--data DIR]\n");
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
        if (!raw) vmem::init({.backend = kind, .name = "test3"});

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
