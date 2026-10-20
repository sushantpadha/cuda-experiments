ctx_overhead: ctx_overhead.cu
	nvcc -std=c++17 -O2 -arch=native $< -o $@ -lcublas -lnvidia-ml
