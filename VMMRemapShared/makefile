NVCC := nvcc
CXXFLAGS := -std=c++17
LDFLAGS := -lcuda -lpthread -lrt

TARGETS := remap_test owner subscriber

.PHONY: all debug clean
all: $(TARGETS)

remap_test: remap_test.cu vmm_remap.cuh common.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

owner: owner.cu ipc_common.h common.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

subscriber: subscriber.cu ipc_common.h common.cuh
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

debug: CXXFLAGS += -g -G -DDEBUG
debug: all

clean:
	rm -f $(TARGETS)
