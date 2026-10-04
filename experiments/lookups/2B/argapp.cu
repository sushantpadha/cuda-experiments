// Test app for argshim.so. Launch patterns with known ground truth:
//   k_direct(A,B)            direct pointer args              -> {A,B}
//   k_struct({C,n,D})        pointers inside a by-value struct -> {C,D}
//   k_indirect(T)            T holds a device pointer to F     -> {T} seen, F missed
//   k_interior(A+1MiB)       interior pointer                  -> {A}
//   cuBLAS sgemm(A,B,C)      library launch path               -> ?
//   N empty launches         per-launch hook overhead
// Build with -cudart shared so LD_PRELOAD can intercept cudaLaunchKernel.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

struct Pair { float *x; int n; float *y; };
__global__ void k_direct(float *a, float *b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) b[i] = a[i] + 1; }
__global__ void k_struct(Pair p) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < p.n) p.y[i] = p.x[i] * 2; }
__global__ void k_indirect(float **t, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) t[0][i] = 3; }
__global__ void k_interior(float *a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] = 4; }
__global__ void k_empty() {}

#define C(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char **argv) {
    long N = argc > 1 ? atol(argv[1]) : 100000;
    const int n = 1 << 22; const size_t B = n * sizeof(float);  // 16 MiB each
    float *A, *Bv, *Cv, *D, *F; float **T;
    C(cudaMalloc(&A, B)); C(cudaMalloc(&Bv, B)); C(cudaMalloc(&Cv, B)); C(cudaMalloc(&D, B));
    C(cudaMalloc(&T, 64)); C(cudaMalloc(&F, B));
    printf("ground truth ids (allocation order): A=0 B=1 C=2 D=3 T=4 F=5\n");
    C(cudaMemcpy(T, &F, sizeof F, cudaMemcpyHostToDevice));
    k_direct<<<n / 256, 256>>>(A, Bv, n);              printf("launch 0 k_direct   expect {0,1}\n");
    k_struct<<<n / 256, 256>>>(Pair{Cv, n, D});         printf("launch 1 k_struct   expect {2,3}\n");
    k_indirect<<<n / 256, 256>>>(T, n);                 printf("launch 2 k_indirect expect {4} seen, 5 touched but invisible\n");
    k_interior<<<1024, 256>>>(A + (1 << 18), 1 << 18);  printf("launch 3 k_interior expect {0}\n");
    C(cudaDeviceSynchronize());
    cublasHandle_t h; cublasCreate(&h); float one = 1, zero = 0; int m = 1024;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &one, A, m, Bv, m, &zero, Cv, m);
    C(cudaDeviceSynchronize()); printf("cublasSgemm(A,B,C) done (touches 0,1,2)\n");
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < N; ++i) k_empty<<<1, 32>>>();
    C(cudaDeviceSynchronize());
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    printf("%ld empty launches: %.3f us/launch (wall, incl. hook if preloaded)\n", N, us / N);
    cublasDestroy(h);
    return 0;
}
