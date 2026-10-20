// pair client: one of two tenants that share one vmemd at the same time. run by tests/daemon/pair_check.py, not by
// hand: make test-pair
//   bin/pair_client ROLE SOCKET      ROLE: a (runs to the end), b (killed by the checker), c (joins after b is gone)
// same line protocol as state_client: "view D H [ALL_D ALL_H]" after each checked step ("local ..." without a broker),
// "WAIT x" blocks until the checker sends a line. the checker releases the clients in the order of its phases
// first config: 24 MiB device, 12 MiB host, 4 MiB context overhead per tenant; chunks of 4 MiB
#include "../utils.cuh"

#include <string>
#include <vector>

using vmem::Link;
using vmem::Loc;
using vmem::Status;

using t::MiB, t::show, t::step, t::wait_for, t::chunk;   // the line protocol, in utils.cuh

// the broker is gone: the next alloc breaks the link and the chunk is made locally
void *broken_alloc(const char *who) {
    void *p = chunk();
    t::expect(p && vmem::get().info().link == Link::Broken, (std::string(who) + ": broker killed: link broken, chunk made locally").c_str());
    return p;
}

void reconnect(const char *who) {
    t::expect(vmem::get().reconnect() == Status::Ok && vmem::get().info().link == Link::Connected,
              (std::string(who) + ": reconnect to the new broker: Ok").c_str());
}

// ---- roles ----

void a() {
    step(0, 0, "a: connected, holds nothing");
    wait_for("hello");
    std::vector<void *> v = {chunk(), chunk(), chunk()};
    t::put(v[0], 4 * MiB, 11);
    step(12, 0, "a: three chunks on device", 20, 0);   // two contexts (8) + 12
    wait_for("b fills");
    // b took the last device chunk and all of the host: both of a's requests are refused and change nothing
    t::expect(chunk() == nullptr, "a: device and host full: alloc refused");
    t::expect(vmem::evict(v[0]) == Status::NoMemory, "a: evict to a full host: refused");
    t::expect(vmem::get().where(v[0]) == Loc::Device && t::intact(v[0], 4 * MiB, 11), "a: refused evict: chunk on device, data intact");
    step(12, 0, "a: refusals change nothing", 24, 12);
    vmem::free(v[0]);
    step(8, 0, "a: frees one device chunk");
    wait_for("b reuses");
    wait_for("b killed");
    // b held 8 MiB device and 12 MiB host; the broker dropped them, so a gets that room
    std::vector<void *> n = {chunk(), chunk(), chunk()};
    step(20, 0, "a: b's device room is reused (three chunks on device)", 24, 0);
    n.push_back(chunk());
    step(20, 4, "a: and its host room", 24, 4);
    for (void *p : n) vmem::free(p);
    step(8, 0, "a: frees them again");
    wait_for("c joined");
    wait_for("kill");
    void *c = broken_alloc("a");
    step(12, 0, "a: local mode: third chunk on device");
    wait_for("restart");
    reconnect("a");
    step(12, 0, "a: counts unchanged by the reconnect");
    wait_for("resynced");
    // second config has no device limit: fill the real VRAM until a chunk lands on host (AllocFail)
    std::vector<void *> big;
    do big.push_back(vmem::alloc(256 * MiB));
    while (big.back() && vmem::get().used(Loc::Host) == 0 && big.size() < 64);
    t::expect(big.back() && vmem::get().used(Loc::Host) == 256 * MiB, "a: the chunk past VRAM landed on host");
    show();
    wait_for("a filled");
    for (void *p : big) vmem::free(p);
    step(12, 0, "a: frees the VRAM fill");
    vmem::free(v[1]);
    vmem::free(v[2]);
    vmem::free(c);
    step(0, 0, "a: frees everything");
}

void b() {
    step(0, 0, "b: connected, holds nothing");
    wait_for("hello");   // released once a holds its three chunks
    // a holds 12 MiB, the two contexts 8: one chunk left on device, so b spills to host after one chunk
    void *d = chunk();
    step(4, 0, "b: one chunk fits on device", 24, 0);
    std::vector<void *> h = {chunk(), chunk(), chunk()};
    t::put(h[0], 4 * MiB, 22);
    step(4, 12, "b: the next three go to host", 24, 12);
    t::expect(chunk() == nullptr, "b: device and host full: alloc refused");
    t::expect(vmem::prefetch(h[0]) == Status::NoMemory, "b: prefetch to a full device: refused");
    t::expect(vmem::get().where(h[0]) == Loc::Host && t::intact(h[0], 4 * MiB, 22), "b: refused prefetch: chunk on host, data intact");
    step(4, 12, "b: refusals change nothing", 24, 12);
    wait_for("a freed");
    void *r = chunk();
    step(8, 12, "b: takes the device room a freed", 24, 12);
    wait_for("hold");   // the checker kills b here, with all its memory held
    (void)d, (void)r;
}

void c() {
    step(0, 0, "c: connected, holds nothing");
    std::vector<void *> v = {chunk(), chunk()};
    step(8, 0, "c: two chunks on device", 24, 0);   // a 4 + 8, c 4 + 8
    wait_for("kill");
    v.push_back(broken_alloc("c"));
    t::expect(vmem::evict(v[0]) == Status::Ok && vmem::get().where(v[0]) == Loc::Host, "c: local mode: evict works");
    step(8, 4, "c: local mode: one chunk on host");
    wait_for("restart");
    reconnect("c");
    step(8, 4, "c: counts unchanged by the reconnect");
    wait_for("resynced");
    wait_for("a filled");
    // a filled the VRAM: c's chunk does not fit on the device either, so c sends its own AllocFail
    void *big = vmem::alloc(256 * MiB);
    t::expect(big && vmem::get().used(Loc::Host) == 260 * MiB, "c: VRAM still full: its 256 MiB chunk landed on host");
    show();
    wait_for("c filled");
    vmem::free(big);
    for (void *p : v) vmem::free(p);
    step(0, 0, "c: frees everything");
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string r = argc == 3 ? argv[1] : "";
    if (r != "a" && r != "b" && r != "c") {
        fprintf(stderr, "usage: pair_client a|b|c SOCKET (run it through tests/daemon/pair_check.py)\n");
        return 2;
    }
    unsetenv("VMEM_BACKEND");
    unsetenv("VMEM_DAEMON");
    try {
        vmem::init({.backend = vmem::Kind::Vmem, .daemon = argv[2], .name = ("pair-" + r).c_str()});
        t::expect(vmem::get().info().link == Link::Connected, "link to vmemd is connected");
        if (r == "a") a();
        else if (r == "b") b();
        else c();
        vmem::shutdown();
    } catch (const vmem::Error &e) {
        printf("FAIL: uncaught %s (%s)\n", e.what(), vmem::status_name(e.status));
        return 2;
    }
    return t::fails != 0;
}
