NVCC     := nvcc
# override per GPU, e.g. make ARCH=-arch=sm_89
ARCH     ?= -arch=native
CXXFLAGS := -std=c++20 -O2 $(ARCH) -Isrc
LDFLAGS  := -lcuda

# src/ the library (header only), src/backends/ one header per backend, tests/ test programs, bin/ binaries
#   make daemon          build bin/vmemd (run: bin/vmemd --config daemon/vmemd.toml [--socket PATH] [--log PATH] [--debug])
#   make test N=1c       build tests/test1.cu and run it on backend c (also 1u 1v 1d 2c 2u 2v 2d; see tests/README.md)
#   make test-unit       all eight of those
#   make stress          4 churn workers (random calls, checked after each) against one vmemd; ARGS="--workers 1 --backend cuda" etc.
#   make test-state      one client against vmemd, scenario by scenario (tests/daemon/state_check.py + bin/state_client)
#   make test-pair       two clients against one vmemd at once (tests/daemon/pair_check.py + bin/pair_client)
#   make test-workers-gemm / -image / -mix   4 worker processes against one vmemd: matrix multiply, image sharpen, or
#                        both in turn (tests/daemon/workers_check.py + bin/worker; ARGS passes options, e.g. --backend raw)
#   make test-daemon     test-state, test-pair, stress, test-workers-gemm, -image, -mix in that order; stops at the first failure
#   make trace-gemm / trace-image / trace-mix   one four-process test under scripts/memtrace.py: VRAM, DRAM and hints over time,
#                        in analysis/WORKLOAD-BACKEND.csv and .events.csv; BACKEND=vmem|vmem-local|cuda|uvm|raw, TRACE_ARGS go to memtrace
#   make trace-all       both workloads on every backend, then plot-traces
#   make plot-traces     analysis/{gemm,image,mix}.png (cuda left out), plus one WORKLOAD-BACKEND.png per run (scripts/memplot.py)
N    ?= 1c
ARGS ?=
DEPS := $(wildcard src/*.h src/*.cuh src/backends/*.h tests/*.cuh)

.PHONY: test test-unit stress test-state test-pair test-workers-gemm test-workers-image test-workers-mix test-daemon trace-gemm trace-image trace-mix trace-all plot-traces daemon clean

bin/test%: tests/test%.cu $(DEPS)
	mkdir -p bin && $(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# the C++ halves of the script-driven daemon tests: state_client, pair_client, worker
bin/%: tests/daemon/%.cu $(DEPS)
	mkdir -p bin && $(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

daemon: bin/vmemd

bin/vmemd: daemon/vmemd.cpp src/protocol.h src/backend.h
	mkdir -p bin && $(CXX) -std=c++20 -O2 -Wall -Wextra -Isrc $< -o $@

# N = test number + backend letter, e.g. 1c: bin/test1 c
TN = $(shell echo $(N) | tr -d a-z)
TB = $(shell echo $(N) | tr -d 0-9)

test: bin/test$(TN) bin/vmemd   # the "d" runs start vmemd
	bin/test$(TN) $(TB) $(ARGS)

test-unit: bin/test1 bin/test2 bin/vmemd
	for n in 1c 1u 1v 1d 2c 2u 2v 2d; do $(MAKE) --no-print-directory test N=$$n || exit 1; done

stress: bin/vmemd bin/worker
	python3 tests/daemon/workers_check.py --workload churn $(ARGS)

test-state: bin/vmemd bin/state_client
	python3 tests/daemon/state_check.py

test-pair: bin/vmemd bin/pair_client
	python3 tests/daemon/pair_check.py

test-workers-gemm: bin/vmemd bin/worker
	python3 tests/daemon/workers_check.py --workload gemm $(ARGS)

test-workers-image: bin/vmemd bin/worker
	python3 tests/daemon/workers_check.py --workload image $(ARGS)

test-workers-mix: bin/vmemd bin/worker
	python3 tests/daemon/workers_check.py --workload mix $(ARGS)

# ---- memory over time (scripts/memtrace.py, scripts/memplot.py) ----
BACKEND ?= vmem
BACKENDS := cuda raw uvm vmem-local vmem
TRACE_ARGS ?=   # e.g. --interval 0.05
# workers print "hint moved|noop|refused UNIX_TIME" into /tmp/vmemd-WORKLOAD-*/workerN.out; record each one's time
OUTS = /tmp/vmemd-$(1)-*/worker*.out
HINTS = --event "hints moved=$(OUTS):^hint moved ([0-9.]+)" --event "hints noop=$(OUTS):^hint noop ([0-9.]+)" \
        --event "hints refused=$(OUTS):^hint refused ([0-9.]+)"

trace-gemm trace-image trace-mix: trace-%: bin/vmemd bin/worker
	python3 ../scripts/memtrace.py $(TRACE_ARGS) -o analysis/$*-$(BACKEND).csv $(call HINTS,$*) -- \
		python3 tests/daemon/workers_check.py --workload $* --backend $(BACKEND) --no-solo $(ARGS)

# every workload on every backend (a run that fails, e.g. cuda out of memory, still leaves its CSV), then the plots
# (the plots leave cuda out: it runs out of memory in every workload)
trace-all: bin/vmemd bin/worker
	for w in gemm image mix; do for b in $(BACKENDS); do $(MAKE) --no-print-directory trace-$$w BACKEND=$$b || true; done; done
	$(MAKE) --no-print-directory plot-traces

plot-traces:
	for w in gemm image mix; do python3 ../scripts/memplot.py -o analysis/$$w.png --title "$$w: four workers at once" \
		--right "hints moved" --right-for "vmem + vmemd,vmem local" \
		"uvm, no hints=analysis/$$w-raw.csv" "uvm=analysis/$$w-uvm.csv" \
		"vmem local=analysis/$$w-vmem-local.csv" "vmem + vmemd=analysis/$$w-vmem.csv"; done
	# one plot per run too, cuda included: analysis/WORKLOAD-BACKEND.png
	for w in gemm image mix; do for b in $(BACKENDS); do \
		case $$b in vmem) l="vmem + vmemd";; vmem-local) l="vmem local";; raw) l="uvm, no hints";; *) l=$$b;; esac; \
		python3 ../scripts/memplot.py -o analysis/$$w-$$b.png --title "$$w: four workers, $$l" \
			--right "hints moved" "$$l=analysis/$$w-$$b.csv"; done; done

test-daemon:
	$(MAKE) test-state && $(MAKE) test-pair && $(MAKE) stress && $(MAKE) test-workers-gemm && $(MAKE) test-workers-image && \
		$(MAKE) test-workers-mix

clean:
	rm -rf bin
