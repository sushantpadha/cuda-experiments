#include<bits/stdc++.h>
#include <cuda/cmath>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                  \
    }                                                                          \
} while (0)

using namespace std;

__global__ void vecAdd(float* A, float* B, float* C) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    C[idx] = A[idx] + B[idx];
}

int main()
{
    float *A, *B, *C;
    
    constexpr int N = 1024;

    CUDA_CHECK( cudaMallocManaged(&A, N*sizeof(float)) );
    CUDA_CHECK( cudaMallocManaged(&B, N*sizeof(float)) );
    CUDA_CHECK( cudaMallocManaged(&C, N*sizeof(float)) );

    for(size_t i = 0; i<N; ++i) A[i] = (float)i;
    for(size_t i = 0; i<N; ++i) B[i] = (float)N-i;

    constexpr int threads = 256;
    constexpr int blocks = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(A, B, C);

    CUDA_CHECK( cudaDeviceSynchronize() );

    bool correct = true;
    for(size_t i = 0; i<N; ++i) {
        if (abs(C[i] - (float)N) > 1e-5) {
            correct = false;
            break;
        }
    }
    if (correct) {
        cout << "Result is correct!\n";
    } else {
        cout << "Result is incorrect!\n";
    }

    CUDA_CHECK( cudaFree(A) );
    CUDA_CHECK( cudaFree(B) );
    CUDA_CHECK( cudaFree(C) );

    return 0;
}