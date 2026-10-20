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
#   make test-workers-gemm / test-workers-image   4 worker processes against one vmemd, matrix multiply or image sharpen (tests/daemon/workers_check.py + bin/worker; ARGS passes options)
#   make test-daemon     test-state, test-pair, test-workers-gemm, test-workers-image in that order; stops at the first failure
#   make trace-gemm / trace-image   one four-process test under scripts/memtrace.py: VRAM, DRAM and hints over time,
#                        in analysis/{gemm,image}-BACKEND.csv; BACKEND=vmem|vmem-local|cuda|uvm, TRACE_ARGS go to memtrace
#   make trace-all       both workloads on every backend, then plot-traces
#   make plot-traces     analysis/gemm.png and image.png from those CSVs (scripts/memplot.py)
N    ?= 1c
ARGS ?=
DEPS := $(wildcard src/*.h src/*.cuh src/backends/*.h tests/*.cuh)

.PHONY: test stress test-state test-pair test-workers-gemm test-workers-image test-daemon trace-gemm trace-image trace-all plot-traces daemon clean

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

test-workers-gemm: bin/vmemd bin/worker
	python3 tests/daemon/workers_check.py --workload gemm $(ARGS)

test-workers-image: bin/vmemd bin/worker
	python3 tests/daemon/workers_check.py --workload image $(ARGS)

# ---- memory over time (scripts/memtrace.py, scripts/memplot.py) ----
BACKEND ?= vmem
BACKENDS := cuda uvm vmem-local vmem
TRACE_ARGS ?=   # e.g. --interval 0.05
# workers print "hint ok" / "hint refused" into /tmp/vmemd-WORKLOAD-*/workerN.out; count them there
HINTS = --count "hints accepted=/tmp/vmemd-$(1)-*/worker*.out:^hint ok" --count "hints refused=/tmp/vmemd-$(1)-*/worker*.out:^hint refused"

trace-gemm trace-image: trace-%: bin/vmemd bin/worker
	python3 ../scripts/memtrace.py $(TRACE_ARGS) -o analysis/$*-$(BACKEND).csv $(call HINTS,$*) -- \
		python3 tests/daemon/workers_check.py --workload $* --backend $(BACKEND) --no-solo $(ARGS)

# every workload on every backend (a run that fails, e.g. cuda out of memory, still leaves its CSV), then the plots
# (the plots leave cuda out: it runs out of memory in both workloads)
trace-all: bin/vmemd bin/worker
	for w in gemm image; do for b in $(BACKENDS); do $(MAKE) --no-print-directory trace-$$w BACKEND=$$b || true; done; done
	$(MAKE) --no-print-directory plot-traces

plot-traces:
	for w in gemm image; do python3 ../scripts/memplot.py -o analysis/$$w.png --title "$$w: four workers at once" \
		--right "hints accepted" --right-for "vmem + vmemd,vmem local" \
		"uvm=analysis/$$w-uvm.csv" \
		"vmem local=analysis/$$w-vmem-local.csv" "vmem + vmemd=analysis/$$w-vmem.csv"; done

test-daemon:
	$(MAKE) test-state && $(MAKE) test-pair && $(MAKE) test-workers-gemm && $(MAKE) test-workers-image

clean:
	rm -rf bin
