// test 2 on the Vmem backend: make test N=2v
#include "test2.cuh"

int main() { return test2("test2v", vmem::Kind::Vmem); }
