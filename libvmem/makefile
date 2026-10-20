NVCC     := nvcc
# override per GPU, e.g. make ARCH=-arch=sm_89
ARCH     ?= -arch=native
CXXFLAGS := -std=c++20 -O2 $(ARCH) -Isrc
LDFLAGS  := -lcuda

# src/ the library (header only), src/backends/ one header per backend, tests/ test programs, bin/ binaries
#   make daemon       build bin/vmemd (run: bin/vmemd [--socket PATH] [--log PATH])
#   make test N=1c    build and run tests/test1c.cu (c cuda, u uvm, v vmem; test1 simple, test2 errors; N=3 is the stress test: make test N=3 ARGS=raw; N=4 vmemd accounting; N=5 vmem backend with vmemd)
N    ?= 1c
ARGS ?=
DEPS := $(wildcard src/*.h src/*.cuh src/backends/*.h tests/*.cuh)

.PHONY: test daemon clean

bin/test%: tests/test%.cu $(DEPS)
	mkdir -p bin && $(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

daemon: bin/vmemd

bin/vmemd: daemon/vmemd.cpp src/protocol.h src/backend.h
	mkdir -p bin && $(CXX) -std=c++20 -O2 -Wall -Wextra -Isrc $< -o $@

bin/test4 bin/test5: bin/vmemd   # tests 4 and 5 run the daemon

test: bin/test$(N)
	bin/test$(N) $(ARGS)

clean:
	rm -rf bin
