// test 1 on the Vmem backend: make test N=1v
#include "test1.cuh"

int main() { return test1(vmem::Kind::Vmem); }
