NVCC := nvcc
TARGET := main
SRC := main.cu

CXXFLAGS := -std=c++17
LDFLAGS := -lcuda

.PHONY: all debug clean

all: $(TARGET)

$(TARGET): $(SRC) vmm_vector.cuh common.cuh
	$(NVCC) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS)

debug: CXXFLAGS += -g -G -DDEBUG
debug: $(TARGET)

clean:
	rm -f $(TARGET)

