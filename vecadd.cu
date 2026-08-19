#include <cstdio>
__global__ void add(int *a, int *b, int *c) { *c = *a + *b; }
int main() {
    int a=2,b=3,c; int *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,sizeof(int)); cudaMalloc(&d_b,sizeof(int)); cudaMalloc(&d_c,sizeof(int));
    cudaMemcpy(d_a,&a,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&b,sizeof(int),cudaMemcpyHostToDevice);
    add<<<1,1>>>(d_a,d_b,d_c);
    cudaMemcpy(&c,d_c,sizeof(int),cudaMemcpyDeviceToHost);
    printf("%d + %d = %d\n", a, b, c);
}
