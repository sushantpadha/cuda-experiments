NVCC     := nvcc
# override per GPU, e.g. make ARCH=-arch=sm_89
ARCH     ?= -arch=native
CXXFLAGS := -std=c++17 -O2 $(ARCH)
LDFLAGS  := -lcuda

TARGETS := smoke main

.PHONY: all debug test clean

all: $(TARGETS)

smoke: smoke.cu vmem.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

main: main.cu vmem.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

debug: CXXFLAGS := -std=c++17 -g -G -DDEBUG $(ARCH)
debug: clean $(TARGETS)

clean:
	rm -f $(TARGETS)
