NVCC := nvcc
# override per GPU, e.g. make ARCH=-arch=sm_89
ARCH ?= -arch=native

vmm_latency: vmm_latency.cu
	$(NVCC) -std=c++17 -O2 $(ARCH) $< -o $@ -lcuda

clean:
	rm -f vmm_latency

.PHONY: clean
