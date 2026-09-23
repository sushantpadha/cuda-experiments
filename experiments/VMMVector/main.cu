#include "common.cuh"
#include "vmm_vector.cuh"
#include <chrono>

void __global__ kernel(float* d_ptr, size_t size) {
    size_t workIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (workIdx < size)
        d_ptr[workIdx] *= 2.0f;
}

void init_driver_state() {
    CU_CHECK(cuInit(0));

    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    CUdevice device;
    cuDeviceGet(&device, 0); // Get handle for device 0

    int numaId = -1;
    CUresult res = cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device);

    if (res == CUDA_SUCCESS && numaId != -1) {
        printf("GPU 0 is closest to Host NUMA ID: %d\n", numaId);
    } else {
        printf("NUMA not supported or unable to get NUMA ID.\n");
    }
}

void print_state(const char* label, VMMVector<float>& vec) {
    printf("%s\n", label);
    printf("  size:          %zu\n", vec.size());
    printf("  buf_size:      %zu\n", vec.buf_size());
    printf("  capacity:      %zu elements\n", vec.capacity());
    printf("  max capacity:  %zu elements\n", vec.max_capacity());
    printf("  chunk size:    %zu MB\n", vec.chunk_size() / 1024 / 1024);
    printf("  retain_last:   %d\n", vec.retain_last());
}

void check_values(VMMVector<float>& vec, size_t n, float multiplier) {
    float* buf = (float*)malloc(vec.buf_size());

    size_t copied_size = vec.copy_to_host(buf);
    assert(copied_size == vec.size() * sizeof(float));

    for (size_t i = 0; i < n; ++i)
        if (!is_equal(buf[i], (float)i * multiplier)) {
            printf("buf[%zu] = %f, expected %f\n", i, buf[i], (float)i * multiplier);
            assert(false);
        }

    free(buf);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <vec_len> [chunk_size_MB]\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, NULL, _IONBF, 0);

    size_t N = argc > 1 ? atoi(argv[1]) : 1'000'000;
    size_t chunk_size = (argc > 2 ? atoi(argv[2]) : 16) * 1024 * 1024;
    size_t count = (argc > 3 ? max(0, atoi(argv[3])) : std::numeric_limits<size_t>::max());

    init_driver_state();

    VMMVector<float> vec(N, chunk_size, 0, true);

    printf("=== CONFIGURATION ===\n");
    printf("sizeof(T):     %zu\n", sizeof(float));
    printf("requested N:   %zu\n", N);
    printf("requested chunk: %zu MB\n", chunk_size / 1024 / 1024);
    print_state("initial", vec);


    printf("\n=== TEST 1 : basic push_back + GPU kernel ===\n");
    count--;
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < N; ++i)
        vec.push_back((float)i);

    print_state("after push_back", vec);

    int threads = 256;
    int blocks = ROUND_UP(vec.size(), threads) / threads;

    kernel<<<blocks, threads>>>(vec.data(), vec.size());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    check_values(vec, N, 2.0f);

    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    printf("GPU values correct\n");
    printf("=== PASSED (%.3f ms) ===\n", elapsed);

    CUmemAllocationProp props;
    CU_CHECK( cuMemGetAllocationPropertiesFromHandle(&props, vec.handles_.back()) );
    printf("  location type: %d\n", props.location.type);
    printf("  location id:   %d\n", props.location.id);
    
    if (count == 0) return 0;

    printf("\n=== TEST 2 : resize down, retain_last = 0 ===\n");
    count--;
    start = std::chrono::steady_clock::now();

    vec.set_retain_last(0);

    size_t old_capacity = vec.capacity();
    size_t new_size = N / 2;

    vec.resize(new_size);

    print_state("after resize", vec);

    assert(vec.size() == new_size);
    assert(vec.capacity() <= old_capacity);

    check_values(vec, new_size, 2.0f);

    elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    printf("values correct\n");
    printf("=== PASSED (%.3f ms) ===\n", elapsed);
    if (count == 0) return 0;


    printf("\n=== TEST 3 : regrow after eager freeing ===\n");
    count--;
    start = std::chrono::steady_clock::now();

    size_t regrow_size = N;

    vec.resize(regrow_size);

    print_state("after regrow", vec);

    assert(vec.size() == regrow_size);
    assert(vec.capacity() >= regrow_size);

    float* buf = (float*)malloc(vec.buf_size());
    size_t copied_size = vec.copy_to_host(buf);
    assert(copied_size == vec.size() * sizeof(float));

    for (size_t i = 0; i < N / 2; ++i)
        assert(is_equal(buf[i], (float)i * 2.0f));

    for (size_t i = N / 2; i < N; ++i)
        assert(is_equal(buf[i], 0.0f));

    free(buf);

    elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    printf("old values preserved, new values zero-initialized\n");
    printf("=== PASSED (%.3f ms) ===\n", elapsed);
    if (count == 0) return 0;


    printf("\n=== TEST 4 : retain_last = 2 ===\n");
    count--;
    start = std::chrono::steady_clock::now();

    vec.set_retain_last(2);

    const size_t chunk_fits = vec.chunk_fits();
    vec.resize(chunk_fits);

    print_state("after resize", vec);

    assert(vec.size() == chunk_fits);

    check_values(vec, min(chunk_fits, new_size), 2.0f);

    elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    printf("values correct\n");
    printf("=== PASSED (%.3f ms) ===\n", elapsed);

    if (count == 0) return 0;

    printf("\n=== TEST 5 : retain_last = -1 ===\n");
    count--;
    start = std::chrono::steady_clock::now();

    size_t capacity_before = vec.capacity();

    vec.set_retain_last(-1);
    vec.resize(0);

    print_state("after resize(0)", vec);

    assert(vec.size() == 0);
    assert(vec.capacity() == capacity_before);

    elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    printf("all chunks retained\n");
    printf("=== PASSED (%.3f ms) ===\n", elapsed);
    if (count == 0) return 0;


    printf("\n=== TEST 6 : regrow with retained chunks ===\n");
    count--;
    start = std::chrono::steady_clock::now();

    vec.resize(N / 2);

    print_state("after regrow", vec);

    assert(vec.size() == N / 2);
    assert(vec.capacity() == capacity_before);

    check_values(vec, N / 2, 0.0f);

    elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    printf("new elements zero-initialized\n");
    printf("=== PASSED (%.3f ms) ===\n", elapsed);

    if (count == 0) return 0;

    printf("\n=== ALL TESTS PASSED ===\n");

    return 0;
}