// test 1 on the Cuda backend: make test N=1c
#include "test1.cuh"

int main() { return test1(vmem::Kind::Cuda); }
