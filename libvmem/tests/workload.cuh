// the workload of the stress tests: tiled matrix multiply, then a sharpen filter on big images
// shared by stress.cu (one process) and daemon/worker.cu (several processes at once)
// memory comes from libvmem, or from plain managed memory when raw is set
#pragma once

#include "libvmem.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
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
// first: index of the first base image, so that workers can use different ones
static int sharpen_phase(bool resident, int count, size_t W, size_t H, const std::vector<Base> &bases, unsigned long long &total,
                         int first = 0) {
    const size_t X = W * H;
    std::vector<unsigned char *> in(count, nullptr);
    int rc = 0;

    auto generate = [&](int i) {
        const Base &b = bases[(first + i) % bases.size()];
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
               bases[(first + i) % bases.size()].name.c_str(), W, H, X >> 20, t1 - t0, t2 - t1, s, bad);
        mem_free(out);
        if (resident) mem_evict(in[i]); else mem_free(in[i]);
        in[i] = resident ? in[i] : nullptr;
    }
    for (unsigned char *p : in)
        if (p) mem_free(p);
    return rc;
}
