#include <cuda_runtime.h>
#include <cstdio>

void printCudaProps(int device = 0) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, device);

    printf("=== Device %d: %s ===\n", device, p.name);
    printf("computeCapability:            %d.%d\n", p.major, p.minor);
    printf("totalGlobalMem:               %zu MB\n", p.totalGlobalMem / (1024 * 1024));
    printf("sharedMemPerBlock:            %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("sharedMemPerMultiprocessor:   %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("regsPerBlock:                 %d\n", p.regsPerBlock);
    printf("regsPerMultiprocessor:        %d\n", p.regsPerMultiprocessor);
    printf("warpSize:                     %d\n", p.warpSize);
    printf("maxThreadsPerBlock:           %d\n", p.maxThreadsPerBlock);
    printf("maxThreadsPerSM:              %d\n", p.maxThreadsPerMultiProcessor);

    printf("maxThreadsDim:                [%d, %d, %d]\n",
           p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("maxGridSize:                  [%d, %d, %d]\n",
           p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

    printf("SM count:                     %d\n", p.multiProcessorCount);
    printf("memoryBusWidth:               %d bits\n", p.memoryBusWidth);
    printf("L2 cache:                     %d KB\n", p.l2CacheSize / 1024);
    printf("constantMemory:               %zu KB\n", p.totalConstMem / 1024);

    printf("concurrentKernels:            %d\n", p.concurrentKernels);
    printf("unifiedAddressing:            %d\n", p.unifiedAddressing);
    printf("managedMemory:                %d\n", p.managedMemory);
    printf("cooperativeLaunch:            %d\n", p.cooperativeLaunch);
    printf("asyncEngineCount:             %d\n", p.asyncEngineCount);
}

int main() {
    printCudaProps();
    return 0;
}