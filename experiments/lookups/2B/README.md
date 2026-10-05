# 2B probes: eviction signals without hardware access bits

Scratch.

Build:

```
nvcc -O2 -std=c++17 -arch=sm_89 -cudart shared argapp.cu -o argapp -lcublas
nvcc -O2 -std=c++17 -arch=sm_89 signals.cu -o signals -lcuda
g++ -O2 -std=c++17 -shared -fPIC argshim.cpp -o argshim.so -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda -ldl
g++ -O2 -std=c++17 -shared -fPIC cupti_shim.cpp -o cupti_shim.so -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcupti -lcuda
```

Runs:

```
ARGSHIM_LOG=1 LD_PRELOAD=$PWD/argshim.so ./argapp 1000 2>&1 | grep -v k_empty           # app kernels found, cuBLAS missing
ARGSHIM_LOG=1 CUDA_INJECTION64_PATH=$PWD/cupti_shim.so ./argapp 1000 2>&1 | grep -v "k_empty\|{ }$"   # cuBLAS found
./argapp 100000 | tail -1                                                                # baseline us/launch
ARGSHIM_NOSCAN=1 CUDA_INJECTION64_PATH=$PWD/cupti_shim.so ./argapp 100000 2>/dev/null | tail -1   # CUPTI only
CUDA_INJECTION64_PATH=$PWD/cupti_shim.so ./argapp 100000 2>&1 | grep -E "us/launch|scan cost"    # CUPTI + scan
nm -D --undefined-only argapp | grep -i launch                                           # shows __cudaLaunchKernel
./signals frac                                                                           # kernel time vs chunks on host
./signals hash                                                                           # hash GB/s, dirty detection
```
