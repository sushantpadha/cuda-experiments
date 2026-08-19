#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

/* macro to index a 1D memory array with 2D indices in row-major order */
/* ld is the leading dimension, i.e. the number of cols in the matrix     */
#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

__global__ void simple_transpose(int m, float *a, float *c) {
    size_t myCol = threadIdx.x + blockIdx.x * blockDim.x; // fastest
    size_t myRow = threadIdx.y + blockIdx.y * blockDim.y;

    if (myRow < m && myCol < m) {
        c[ INDX(myCol, myRow, m) ] = a[ INDX(myRow, myCol, m) ];
    }
    return;
}

__global__ void smem_transpose(int m, float *a, float *c) {
    // note the block accesses memory tile of the form `float[ THREADS_PER_BLOCK_Y ][ THREADS_PER_BLOCK_X ]`
    // we create smem to store transpose of the tile in smem to ensure coalesced global write
    __shared__ float smemArray[ THREADS_PER_BLOCK_X ][ THREADS_PER_BLOCK_Y ];

    size_t tileX = blockIdx.x * blockDim.x;
    size_t tileY = blockIdx.y * blockDim.y;

    size_t myCol = threadIdx.x + tileX; // fastest
    size_t myRow = threadIdx.y + tileY;    

    if (myRow < m && myCol < m) {
        // coalesced read from global
        // uncoalesced write (might be hitting same banks which is bad) but to smem so its fine
        smemArray[ threadIdx.x ][ threadIdx.y ] = a[ INDX(myRow, myCol, m) ];
    }

    // note: disabling this reduces by ~1 ms (7% reduction) at size=16384
    __syncthreads();

    if (myRow < m && myCol < m) {
        // coalesced read from smem and write to global!
        c[ INDX(tileX + threadIdx.y, tileY + threadIdx.x, m) ] = smemArray[ threadIdx.y ][ threadIdx.x ];
    }

    return;
}

__global__ void smem_transpose_banked(int m, float *a, float *c) {
    // notice THREADS_PER_BLOCK_Y * sizeof(float) % 32 == 0, so all threads access the same bank in the write phase
    // we can avoid this by simply making each array odd sized!
    __shared__ float smemArray[ THREADS_PER_BLOCK_X ][ THREADS_PER_BLOCK_Y + 1 ];

    size_t tileX = blockIdx.x * blockDim.x;
    size_t tileY = blockIdx.y * blockDim.y;

    size_t myCol = threadIdx.x + tileX; // fastest
    size_t myRow = threadIdx.y + tileY;    

    if (myRow < m && myCol < m) {
        // coalesced read from global
        // uncoalesced write (might be hitting same banks which is bad) but to smem so its fine
        smemArray[ threadIdx.x ][ threadIdx.y ] = a[ INDX(myRow, myCol, m) ];
    }

    // note: disabling this reduces by ~1 ms (7% reduction) at size=16384
    __syncthreads();

    if (myRow < m && myCol < m) {
        // coalesced read from smem and write to global!
        c[ INDX(tileX + threadIdx.y, tileY + threadIdx.x, m) ] = smemArray[ threadIdx.y ][ threadIdx.x ];
    }

    return;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? std::atoi(argv[1]) : 4096;

    size_t bytes = (size_t)m * m * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    for (size_t i = 0; i < (size_t)m * m; ++i)
        h_a[i] = (float)(i % 1000);

    for (int r = 0; r < m; ++r)
        for (int c = 0; c < m; ++c)
            h_ref[INDX(c, r, m)] = h_a[INDX(r, c, m)];

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // dont want to see cold cache behaviour
    simple_transpose<<<grid, block>>>(m, d_a, d_c);
    cudaDeviceSynchronize();

    for (int run = 0; run < 5; ++run) {
        cudaMemset(d_c, 0, bytes);

        cudaEventRecord(start);
        simple_transpose<<<grid, block>>>(m, d_a, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for (size_t i = 0; i < (size_t)m * m; ++i) {
            if (h_c[i] != h_ref[i]) {
                correct = false;
                break;
            }
        }

        printf("simple_transpose run %d: %.3f ms (%s)\n",
               run + 1, ms, correct ? "correct" : "WRONG");
    }

    for (int run = 0; run < 5; ++run) {
        cudaMemset(d_c, 0, bytes);

        cudaEventRecord(start);
        smem_transpose<<<grid, block>>>(m, d_a, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for (size_t i = 0; i < (size_t)m * m; ++i) {
            if (h_c[i] != h_ref[i]) {
                correct = false;
                break;
            }
        }

        printf("smem_transpose run %d: %.3f ms (%s)\n",
               run + 1, ms, correct ? "correct" : "WRONG");
    }

    for (int run = 0; run < 5; ++run) {
        cudaMemset(d_c, 0, bytes);

        cudaEventRecord(start);
        smem_transpose_banked<<<grid, block>>>(m, d_a, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for (size_t i = 0; i < (size_t)m * m; ++i) {
            if (h_c[i] != h_ref[i]) {
                correct = false;
                break;
            }
        }

        printf("smem_transpose_banked run %d: %.3f ms (%s)\n",
               run + 1, ms, correct ? "correct" : "WRONG");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_c);

    free(h_a);
    free(h_c);
    free(h_ref);

    return 0;
}