// test 5: the vmem backend as a vmemd client: make test N=5
// starts bin/vmemd with small limits and checks that the broker, not the device, decides where chunks go
// - no broker at init: local mode
// - alloc, evict and prefetch follow the broker's limits; data survives every move
// - broker killed: link Broken, the app goes on locally; reconnect() resyncs the live chunks
// - broker hung: the 2 s receive timeout breaks the link instead of hanging the app
// - device cuMemCreate fails although the broker said device: the chunk goes to host and the broker hears AllocFail
#include "utils.cuh"
#include "daemon.cuh"

#include <fstream>
#include <sstream>
#include <vector>

using vmem::Link;
using vmem::Loc;
using vmem::Status;

constexpr size_t MiB = 1 << 20;

void init() { vmem::init({.backend = vmem::Kind::Vmem, .daemon = SOCK, .name = "test5"}); }
Link link() { return vmem::get().info().link; }

// 32 MiB chunk filled with pattern k; nullptr if refused
void *chunk(unsigned k) {
    void *p = vmem::alloc(32 * MiB);
    if (p) t::put(p, 32 * MiB, k);
    return p;
}

bool log_has(const char *s) {
    std::stringstream ss;
    ss << std::ifstream(LOG).rdbuf();
    return ss.str().find(s) != std::string::npos;
}

// limits: 64 MiB VRAM, 32 MiB DRAM, no context overhead
void limit_tests() {
    printf("no broker: init runs in local mode\n");
    init();
    t::expect(link() == Link::Local, "link is local");
    void *p = chunk(1);
    t::expect(p && vmem::get().where(p) == Loc::Device, "chunk on device");
    vmem::free(p);
    vmem::shutdown();

    pid_t pid = start("vram_limit_mib = 64\ndram_limit_mib = 32\n");
    if (pid < 0) return t::expect(false, "daemon starts with 64/32 MiB limits");
    init();
    t::expect(link() == Link::Connected, "broker running: link is connected");

    printf("alloc: the broker's limits decide, not free VRAM\n");
    void *a = chunk(1), *b = chunk(2), *c = chunk(3);
    auto &al = vmem::get();
    t::expect(a && b && al.where(a) == Loc::Device && al.where(b) == Loc::Device, "two chunks on device (64/64)");
    t::expect(c && al.where(c) == Loc::Host, "third chunk on host (32/32)");
    t::expect(c && t::intact(c, 32 * MiB, 3), "host chunk keeps its data");
    t::expect(vmem::alloc(3 * MiB) == nullptr, "both limits reached: alloc returns nullptr");

    printf("moves: refused without room, granted after a free\n");
    t::expect(vmem::prefetch(c) == Status::NoMemory && al.where(c) == Loc::Host, "prefetch to a full device: NoMemory");
    t::expect(vmem::evict(a) == Status::NoMemory && al.where(a) == Loc::Device, "evict to a full host: NoMemory");
    vmem::free(c);   // host 0/32
    t::expect(vmem::evict(a) == Status::Ok && al.where(a) == Loc::Host, "evict after the free: Ok");
    t::expect(t::intact(a, 32 * MiB, 1), "evicted chunk keeps its data");
    c = chunk(3);
    t::expect(c && al.where(c) == Loc::Device, "the freed device room is reused (64/64)");

    printf("broker killed: the app goes on locally, reconnect() resyncs\n");
    stop(pid);
    vmem::free(b);   // the Free cannot be sent: link breaks
    t::expect(link() == Link::Broken, "link is broken after the broker died");
    b = chunk(2);
    t::expect(b && al.where(b) == Loc::Device, "local mode: new chunk on device");
    t::expect(al.reconnect() == Status::Disconnected && link() == Link::Broken, "reconnect without a broker: Disconnected");
    pid = start("vram_limit_mib = 64\ndram_limit_mib = 32\n");
    if (pid < 0) return t::expect(false, "daemon restarts");
    t::expect(al.reconnect() == Status::Ok && link() == Link::Connected, "reconnect to the new broker: Ok");
    t::expect(vmem::alloc(3 * MiB) == nullptr, "resynced chunks fill both limits: alloc returns nullptr");
    vmem::free(b);   // device 32/64
    void *d = vmem::alloc(3 * MiB);
    t::expect(d && al.where(d) == Loc::Device, "after a free the device has room again");
    vmem::free(d);
    t::expect(t::intact(a, 32 * MiB, 1) && t::intact(c, 32 * MiB, 3), "data intact through the restart");

    printf("broker hung: the receive timeout breaks the link\n");
    kill(pid, SIGSTOP);
    d = vmem::alloc(3 * MiB);
    t::expect(d && link() == Link::Broken, "alloc returns after the timeout, served locally");
    kill(pid, SIGCONT);
    vmem::free(a);
    vmem::free(c);
    vmem::free(d);
    vmem::shutdown();
    stop(pid);
}

// no limits: the broker says device until real VRAM runs out
void alloc_fail_test() {
    printf("alloc_fail: the device is full although the broker said device\n");
    pid_t pid = start("# no limits\n");
    if (pid < 0) return t::expect(false, "daemon starts without limits");
    init();
    std::vector<void *> v;
    while (v.size() < 64) {   // 64 x 256 MiB = 16 GiB, well past VRAM
        void *p = vmem::alloc(256 * MiB);
        if (!p) break;
        v.push_back(p);
        if (vmem::get().where(p) == Loc::Host) break;
    }
    t::expect(!v.empty() && vmem::get().where(v.back()) == Loc::Host, "the chunk past VRAM lands on host");
    usleep(50000);   // AllocFail has no reply
    t::expect(log_has("alloc_fail: pid"), "the broker logged the AllocFail");
    for (void *p : v) vmem::free(p);
    vmem::shutdown();
    stop(pid);
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (access("bin/vmemd", X_OK)) {
        printf("FAIL: bin/vmemd missing (make daemon)\n");
        return 2;
    }
    unsetenv("VMEM_BACKEND");
    unsetenv("VMEM_DAEMON");   // the test picks its own socket
    unlink(LOG);
    unlink(SOCK);
    try {
        limit_tests();
        alloc_fail_test();
    } catch (const vmem::Error &e) {
        printf("FAIL: uncaught %s (%s)\n", e.what(), vmem::status_name(e.status));
        return 2;
    }
    unlink(CONF);
    printf("daemon log: %s\n", LOG);
    printf(t::fails ? "FAIL: %d check(s)\n" : "PASS\n", t::fails);
    return t::fails != 0;
}
