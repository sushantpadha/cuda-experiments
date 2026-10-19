NVCC     := nvcc
# override per GPU, e.g. make ARCH=-arch=sm_89
ARCH     ?= -arch=native
CXXFLAGS := -std=c++20 -O2 $(ARCH) -Isrc -I../primitives/src
LDFLAGS  := -lcuda

# src/ the library (header only), src/backends/ one header per backend, tests/ test programs, bin/ binaries
#   make test N=1     build and run tests/test1.cu
N    ?= 1
ARGS ?=
DEPS := $(wildcard src/*.h src/*.cuh src/backends/*.h) ../primitives/src/pattern.cuh

.PHONY: test clean

bin/test%: tests/test%.cu $(DEPS)
	mkdir -p bin && $(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

test: bin/test$(N)
	bin/test$(N) $(ARGS)

clean:
	rm -rf bin
