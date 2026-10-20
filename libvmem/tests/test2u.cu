// test 2 on the Uvm backend: make test N=2u
#include "test2.cuh"

int main() { return test2("test2u", vmem::Kind::Uvm); }
