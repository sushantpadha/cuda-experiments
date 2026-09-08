#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/cmath>
#include <vector>

// ! cuda runtime calls return cudaError_t
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

// ! cuda driver calls return CUresult
#define CU_CHECK(call)                                                       \
do {                                                                         \
    CUresult err = (call);                                                   \
    if (err != CUDA_SUCCESS) {                                               \
        const char *name, *msg;                                              \
        cuGetErrorName(err, &name);                                          \
        cuGetErrorString(err, &msg);                                         \
        fprintf(stderr, "CUDA Driver error at %s:%d: %s (%s)\n",             \
                __FILE__, __LINE__, msg, name);                              \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)

#ifdef DEBUG
#define DPRINT(fmt, ...) \
    fprintf(stderr, "[DBG] " fmt "\n", ##__VA_ARGS__)
#else
#define DPRINT(fmt, ...)
#endif

inline size_t ROUND_UP(size_t x, size_t granularity) {
    return (x + granularity - 1) / granularity * granularity;
}

inline bool is_equal(float a, float b, float tol = 1e-5) {
    return abs(a - b) < tol;
}