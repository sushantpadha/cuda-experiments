// Test: fill on device -> remap chunks to host (VA unchanged, contents kept)
//       -> remap back -> run kernel -> verify.
//   ./remap_test [n_elems] [chunk_MB]

#include "common.cuh"
#include "vmm_remap.cuh"
#include <chrono>
#include <cassert>

__global__ void dbl(float* p, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] *= 2.0f;
}

static void init_driver() {
    CU_CHECK( cuInit(0) );
    CUdevice d; CU_CHECK( cuDeviceGet(&d, 0) );
    CUcontext c; CU_CHECK( cuDevicePrimaryCtxRetain(&c, d) );
    CU_CHECK( cuCtxSetCurrent(c) );
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    size_t n     = argc > 1 ? strtoull(argv[1], nullptr, 10) : (size_t)1 << 20;
    size_t chunk = (argc > 2 ? strtoull(argv[2], nullptr, 10) : 2) << 20;

    init_driver();
    VMMRemapVector<float> vec(n, chunk);
    for (size_t i = 0; i < n; ++i) vec.push_back((float)i);

    int T = 256, B = (int)((vec.size() + T - 1) / T);
    dbl<<<B, T>>>(vec.data(), vec.size());
    CUDA_CHECK( cudaDeviceSynchronize() );

    std::vector<float> buf(n);
    vec.copy_to_host(buf.data());
    for (size_t i = 0; i < n; ++i) assert(is_equal(buf[i], (float)i * 2.0f));
    printf("filled + doubled: %zu elems across %zu chunks\n", n, vec.n_chunks());

    CUdeviceptr va0 = vec.d_ptr;

    // --- remap every chunk to host ---
    auto t = std::chrono::steady_clock::now();
    for (size_t c = 0; c < vec.n_chunks(); ++c) vec.remap(c, /*to_host=*/true);
    double t_to_host = ms_since(t);

    assert(vec.d_ptr == va0);                       // VA unchanged
    for (size_t c = 0; c < vec.n_chunks(); ++c) assert(vec.chunk_on_host(c));

    std::fill(buf.begin(), buf.end(), -1.0f);
    vec.copy_to_host(buf.data());
    for (size_t i = 0; i < n; ++i) assert(is_equal(buf[i], (float)i * 2.0f));
    printf("remapped %zu chunks device->host in %.3f ms, VA + contents intact\n",
           vec.n_chunks(), t_to_host);

    // --- remap back to device, then double again on the GPU ---
    t = std::chrono::steady_clock::now();
    for (size_t c = 0; c < vec.n_chunks(); ++c) vec.remap(c, /*to_host=*/false);
    double t_to_dev = ms_since(t);

    assert(vec.d_ptr == va0);
    dbl<<<B, T>>>(vec.data(), vec.size());
    CUDA_CHECK( cudaDeviceSynchronize() );
    vec.copy_to_host(buf.data());
    for (size_t i = 0; i < n; ++i) assert(is_equal(buf[i], (float)i * 4.0f));
    printf("remapped %zu chunks host->device in %.3f ms, kernel ran, values = 4*i\n",
           vec.n_chunks(), t_to_dev);

    printf("=== remap_test PASSED ===\n");
    return 0;
}
