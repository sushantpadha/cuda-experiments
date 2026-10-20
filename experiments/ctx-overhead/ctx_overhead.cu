// per-process CUDA context overhead (tracker 1Bii): device-wide used VRAM, read through NVML by a parent
// that never creates a context, while a child steps through cudaFree(0), a first kernel, and cuBLAS
//   ./ctx_overhead [reps]                       CUDA_MODULE_LOADING=EAGER ./ctx_overhead for the eager case
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

__global__ void touch(int *p) { *p = 1; }

const char *STEPS[] = {"cudaFree(0)", "first kernel", "cublasCreate", "first sgemm"};
constexpr int NSTEPS = 4;

// ---- child: one step per byte read from the parent, one byte back when done ----

static void child(int in, int out) {
    char b;
    int *d = nullptr;
    cublasHandle_t h;
    float *m = nullptr, one = 1.f;
    for (int s = 0; s < NSTEPS && read(in, &b, 1) == 1; s++) {
        if (s == 0) cudaFree(0);
        if (s == 1) cudaMalloc(&d, 4), touch<<<1, 1>>>(d);   // 4 bytes, one 2 MiB page at most
        if (s == 2) cublasCreate(&h);
        if (s == 3) cudaMalloc(&m, 3 * 64 * 64 * 4), cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64, &one, m, 64,
                                                                m + 4096, 64, &one, m + 8192, 64);
        b = cudaDeviceSynchronize() == cudaSuccess ? 1 : 0;
        if (write(out, &b, 1) != 1) break;
    }
    while (read(in, &b, 1) == 1) {}   // stay alive until the parent has read the last step
    _exit(0);
}

// ---- parent ----

static unsigned long long used(nvmlDevice_t dev) {
    usleep(200000);   // let the driver settle
    nvmlMemory_t m;
    nvmlDeviceGetMemoryInfo(dev, &m);
    return m.used;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    nvmlDevice_t dev;
    if (nvmlInit() != NVML_SUCCESS || nvmlDeviceGetHandleByIndex(0, &dev) != NVML_SUCCESS) {
        fprintf(stderr, "nvml init failed\n");
        return 1;
    }
    const char *ml = getenv("CUDA_MODULE_LOADING");
    printf("module loading: %s\n", ml ? ml : "default (lazy)");
    printf("rep,step,delta_mib\n");
    for (int r = 0; r < reps; r++) {
        int to[2], from[2];
        if (pipe(to) || pipe(from)) return 1;
        unsigned long long base = used(dev);
        pid_t pid = fork();
        if (pid == 0) close(to[1]), close(from[0]), child(to[0], from[1]);
        for (int s = 0; s < NSTEPS; s++) {
            char b = 0;
            if (write(to[1], &b, 1) != 1 || read(from[0], &b, 1) != 1 || !b) {
                fprintf(stderr, "child failed at %s\n", STEPS[s]);
                return 1;
            }
            printf("%d,%s,%.1f\n", r, STEPS[s], ((long long)used(dev) - (long long)base) / 1048576.0);
        }
        close(to[1]);
        waitpid(pid, nullptr, 0);
        close(to[0]), close(from[0]), close(from[1]);
        printf("%d,after exit,%.1f\n", r, ((long long)used(dev) - (long long)base) / 1048576.0);
    }
    nvmlShutdown();
}
