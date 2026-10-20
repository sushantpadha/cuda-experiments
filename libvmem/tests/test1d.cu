// test 1 on the Vmem backend with a vmemd broker: make test N=1d
// starts bin/vmemd with room for the whole test, then runs the same body as test1c, test1u and test1v
#include "daemon.cuh"
#include "test1.cuh"

int main() {
    unlink(LOG);
    pid_t pid = start("vram_limit_mib = 5000\ndram_limit_mib = 4000\n");
    if (pid < 0) {
        printf("FAIL: vmemd did not start\n");
        return 2;
    }
    int rc = test1("test1d", vmem::Kind::Vmem, SOCK);
    stop(pid);
    printf("vmemd log: %s\n", LOG);
    return rc;
}
