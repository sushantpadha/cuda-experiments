#include "common.cuh"
#include "vmm_vector.cuh"

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
        assert(is_equal(buf[i], (float)i * multiplier));

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

    init_driver_state();

    VMMVector<float> vec(N, chunk_size);

    printf("=== CONFIGURATION ===\n");
    printf("sizeof(T):     %zu\n", sizeof(float));
    printf("requested N:   %zu\n", N);
    printf("requested chunk: %zu MB\n", chunk_size / 1024 / 1024);
    print_state("initial", vec);


    printf("\n=== TEST 1 : basic push_back + GPU kernel ===\n");

    for (size_t i = 0; i < N; ++i)
        vec.push_back((float)i);

    print_state("after push_back", vec);

    int threads = 256;
    int blocks = ROUND_UP(vec.size(), threads) / threads;

    kernel<<<blocks, threads>>>(vec.data(), vec.size());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    check_values(vec, N, 2.0f);

    printf("GPU values correct\n");
    printf("=== PASSED ===\n");


    printf("\n=== TEST 2 : resize down, retain_last = 0 ===\n");

    vec.set_retain_last(0);

    size_t old_capacity = vec.capacity();
    size_t new_size = N / 2;

    vec.resize(new_size);

    print_state("after resize", vec);

    assert(vec.size() == new_size);
    assert(vec.capacity() <= old_capacity);

    check_values(vec, new_size, 2.0f);

    printf("values correct\n");
    printf("=== PASSED ===\n");


    printf("\n=== TEST 3 : regrow after eager freeing ===\n");

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

    printf("old values preserved, new values zero-initialized\n");
    printf("=== PASSED ===\n");


    printf("\n=== TEST 4 : retain_last = 2 ===\n");

    vec.set_retain_last(2);

    const size_t chunk_fits = vec.chunk_fits();
    vec.resize(chunk_fits);

    print_state("after resize", vec);

    assert(vec.size() == chunk_fits);

    check_values(vec, chunk_fits, 2.0f);

    printf("values correct\n");
    printf("=== PASSED ===\n");


    printf("\n=== TEST 5 : retain_last = -1 ===\n");

    size_t capacity_before = vec.capacity();

    vec.set_retain_last(-1);
    vec.resize(0);

    print_state("after resize(0)", vec);

    assert(vec.size() == 0);
    assert(vec.capacity() == capacity_before);

    printf("all chunks retained\n");
    printf("=== PASSED ===\n");


    printf("\n=== TEST 6 : regrow with retained chunks ===\n");

    vec.resize(N / 2);

    print_state("after regrow", vec);

    assert(vec.size() == N / 2);
    assert(vec.capacity() == capacity_before);

    check_values(vec, N / 2, 0.0f);

    printf("new elements zero-initialized\n");
    printf("=== PASSED ===\n");


    printf("\n=== ALL TESTS PASSED ===\n");

    return 0;
}