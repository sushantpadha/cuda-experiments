NVCC     := nvcc
# override per GPU, e.g. make ARCH=-arch=sm_89
ARCH     ?= -arch=native
CXXFLAGS := -std=c++20 -O2 $(ARCH) -Isrc
LDFLAGS  := -lcuda

# src/ sources, bin/ binaries, tests/ test programs, actions/ .vm files for the runner
#   make              bin/runner
#   make test N=1     build and run tests/test1.cu (N=3 ARGS="12 24": timing test)
#   make run N=1      run actions/test1.vm with bin/runner
#   make fuzz         tests/fuzz.py against bin/runner
# extra flags via ARGS: make test N=2 ARGS=-d, make run N=2 ARGS=-v, make fuzz ARGS="-n 200"
N    ?= 1
ARGS ?=
DEPS := src/vmem.cuh src/pattern.cuh
BUILD = mkdir -p bin && $(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: all runner test run fuzz debug clean

all: runner
runner: bin/runner

bin/runner: src/runner/main.cu $(wildcard src/runner/*.cuh) $(DEPS)
	$(BUILD)
bin/test%: tests/test%.cu $(DEPS)
	$(BUILD)

test: bin/test$(N)
	bin/test$(N) $(ARGS)

run: bin/runner
	bin/runner $(ARGS) actions/test$(N).vm

fuzz: bin/runner
	tests/fuzz.py --runner bin/runner $(ARGS)

debug: CXXFLAGS := -std=c++20 -g -G -DDEBUG $(ARCH) -Isrc
debug: clean all

clean:
	rm -rf bin
