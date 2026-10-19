// test 2 on the Cuda backend: make test N=2c
#include "test2.cuh"

int main() { return test2(vmem::Kind::Cuda); }
