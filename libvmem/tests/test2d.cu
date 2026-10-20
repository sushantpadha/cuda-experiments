// test 2 on the Vmem backend with a vmemd broker: make test N=2d
// the limits are small enough that the out-of-memory step hits the broker's refusal, not the pinned memory limit of the machine
#include "daemon.cuh"
#include "test2.cuh"

int main() {
    unlink(LOG);
    pid_t pid = start("vram_limit_mib = 5000\ndram_limit_mib = 3000\n");
    if (pid < 0) {
        printf("FAIL: vmemd did not start\n");
        return 2;
    }
    int rc = test2("test2d", vmem::Kind::Vmem, SOCK);
    stop(pid);
    printf("vmemd log: %s\n", LOG);
    return rc;
}
