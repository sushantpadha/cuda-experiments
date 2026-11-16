
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CU_CHECK(call) do {                                      \
    CUresult e = (call);                                          \
    if (e != CUDA_SUCCESS) {                                      \
        const char *s;                                            \
        cuGetErrorString(e, &s);                                  \
        fprintf(stderr, "%s failed: %s\n", #call, s);             \
        exit(2);                                                  \
    }                                                             \
} while (0)

#define CUDA_CHECK(call) do {                                    \
    cudaError_t e = (call);                                       \
    if (e != cudaSuccess) {                                       \
        fprintf(stderr, "%s failed: %s\n", #call,                 \
                cudaGetErrorString(e));                            \
        exit(2);                                                  \
    }                                                             \
} while (0)

__global__ void write_kernel(volatile unsigned int *p) {
    *p = 0x12345678;
}

__global__ void read_kernel(const volatile unsigned int *p,
                            unsigned int *out) {
    out[0] = *p;
}

static void check_kernel(const char *test) {
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess)
        e = cudaDeviceSynchronize();

    if (e != cudaSuccess) {
        fprintf(stderr, "[%s] GPU access failed: %s (%d)\n",
                test, cudaGetErrorString(e), (int)e);
        fprintf(stderr, "Treat this context as unusable; exiting.\n");
        exit(1);
    }

    printf("[%s] GPU access succeeded\n", test);
}

int main(int argc, char **argv) {
    if (argc != 2 ||
        (strcmp(argv[1], "readonly-write") != 0 &&
         strcmp(argv[1], "none-read") != 0 &&
         strcmp(argv[1], "unmapped-read") != 0)) {
        fprintf(stderr,
            "Usage: %s readonly-write|none-read|unmapped-read\n",
            argv[0]);
        return 2;
    }

    // Use the CUDA runtime's primary context for both APIs.
    CUDA_CHECK(cudaSetDevice(0));
    CU_CHECK(cuInit(0));

    CUdevice dev;
    CUcontext ctx;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CU_CHECK(cuCtxGetCurrent(&ctx));

    if (!ctx) {
        fprintf(stderr, "No current CUDA context\n");
        return 2;
    }

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;

    size_t granularity;
    CU_CHECK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, granularity, &prop, 0));

    CUdeviceptr va;
    CU_CHECK(cuMemAddressReserve(&va, granularity, granularity, 0, 0));
    CU_CHECK(cuMemMap(va, granularity, 0, handle, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CU_CHECK(cuMemSetAccess(va, granularity, &access, 1));

    // Initialize the backing memory while it is writable.
    write_kernel<<<1, 1>>>(
        reinterpret_cast<volatile unsigned int *>(va));
    check_kernel("initialize");

    unsigned int *out;
    CUDA_CHECK(cudaMalloc(&out, sizeof(*out)));

    if (strcmp(argv[1], "readonly-write") == 0) {
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        CU_CHECK(cuMemSetAccess(va, granularity, &access, 1));

        printf("Testing GPU write to read-only VMM mapping...\n");
        write_kernel<<<1, 1>>>(
            reinterpret_cast<volatile unsigned int *>(va));
        check_kernel("readonly-write");

    } else if (strcmp(argv[1], "none-read") == 0) {
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
        CU_CHECK(cuMemSetAccess(va, granularity, &access, 1));

        printf("Testing GPU read from PROT_NONE mapping...\n");
        read_kernel<<<1, 1>>>(
            reinterpret_cast<const volatile unsigned int *>(va), out);
        check_kernel("none-read");

    } else {
        CU_CHECK(cuMemUnmap(va, granularity));

        printf("Testing GPU read from reserved but unmapped VA...\n");
        read_kernel<<<1, 1>>>(
            reinterpret_cast<const volatile unsigned int *>(va), out);
        check_kernel("unmapped-read");
    }

    // Reached only if the tested access succeeded.
    CUDA_CHECK(cudaFree(out));
    CU_CHECK(cuMemAddressFree(va, granularity));
    CU_CHECK(cuMemRelease(handle));

    return 0;
}


