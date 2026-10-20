// test 1 on the Vmem backend: make test N=1v
#include "test1.cuh"

int main() { return test1("test1v", vmem::Kind::Vmem); }
