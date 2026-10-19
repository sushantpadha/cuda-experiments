// test 1 on the Uvm backend: make test N=1u
#include "test1.cuh"

int main() { return test1(vmem::Kind::Uvm); }
