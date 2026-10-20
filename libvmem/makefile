NVCC     := nvcc
# override per GPU, e.g. make ARCH=-arch=sm_89
ARCH     ?= -arch=native
CXXFLAGS := -std=c++20 -O2 $(ARCH) -Isrc
LDFLAGS  := -lcuda

# src/ the library (header only), src/backends/ one header per backend, tests/ test programs, bin/ binaries
#   make daemon          build bin/vmemd (run: bin/vmemd --config daemon/vmemd.toml [--socket PATH] [--log PATH] [--debug])
#   make test N=1c       build and run tests/test1c.cu (also 1u 1v 1d 2c 2u 2v 2d; see tests/README.md)
#   make stress ARGS=raw stress test on one process (modes raw, cuda, uvm, vmemd; flags --resident --images N --n MATRIX_N)
#   make test-state      one client against vmemd, scenario by scenario (tests/daemon/state_check.py + bin/state_client)
#   make test-pair       two clients against one vmemd at once (tests/daemon/pair_check.py + bin/pair_client)
#   make test-workers    4 worker processes against one vmemd (tests/daemon/workers_check.py + bin/worker; ARGS passes options)
#   make test-daemon     test-state, test-pair, test-workers in that order; stops at the first failure
N    ?= 1c
ARGS ?=
DEPS := $(wildcard src/*.h src/*.cuh src/backends/*.h tests/*.cuh)

.PHONY: test stress test-state test-pair test-workers test-daemon daemon clean

bin/test%: tests/test%.cu $(DEPS)
	mkdir -p bin && $(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

bin/stress: tests/stress.cu $(DEPS)
	mkdir -p bin && $(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# the C++ halves of the script-driven daemon tests: state_client, pair_client, worker
bin/%: tests/daemon/%.cu $(DEPS)
	mkdir -p bin && $(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

daemon: bin/vmemd

bin/vmemd: daemon/vmemd.cpp src/protocol.h src/backend.h
	mkdir -p bin && $(CXX) -std=c++20 -O2 -Wall -Wextra -Isrc $< -o $@

bin/test1d bin/test2d: bin/vmemd   # these start the daemon

test: bin/test$(N)
	bin/test$(N) $(ARGS)

stress: bin/stress
	bin/stress $(ARGS)

test-state: bin/vmemd bin/state_client
	python3 tests/daemon/state_check.py

test-pair: bin/vmemd bin/pair_client
	python3 tests/daemon/pair_check.py

test-workers: bin/vmemd bin/worker
	python3 tests/daemon/workers_check.py $(ARGS)

test-daemon:
	$(MAKE) test-state && $(MAKE) test-pair && $(MAKE) test-workers

clean:
	rm -rf bin
