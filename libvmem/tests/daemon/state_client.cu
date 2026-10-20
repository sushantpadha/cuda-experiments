// state client: one tenant of the vmemd state test. run by tests/daemon/state_check.py, not by hand: make test-state
//   bin/state_client SCENARIO SOCKET
// each step checks this process's own device and host MiB, then prints "view D H" (or "local D H" while it has no
// broker). the checker compares the views with the state lines in the vmemd log. "WAIT x" asks the checker for
// something (restart, stop or resume the broker, run another client) and blocks until it sends a line on stdin
// scenarios: simple, multi, resync, allocfail, hold, fill, hang, nobroker, autoreconnect
#include "../utils.cuh"

#include <chrono>
#include <string>
#include <vector>

using vmem::Link;
using vmem::Loc;
using vmem::Status;

using t::MiB, t::show, t::step, t::wait_for, t::chunk;   // the line protocol, in utils.cuh

// ---- scenarios ----

// one tenant, one call at a time. the broker adds 4 MiB of context overhead, which the checker knows about
void simple() {
    void *a = chunk();
    step(4, 0, "alloc 4 MiB: device");
    void *b = chunk();
    step(8, 0, "alloc 4 MiB: device");
    vmem::free(a);
    step(4, 0, "free the first");
    vmem::free(b);
    step(0, 0, "free the second");
    // small objects share one 2 MiB chunk: the broker sees the chunk, never the objects
    void *s1 = vmem::alloc(100);
    step(2, 0, "small object: one 2 MiB chunk on device");
    void *s2 = vmem::alloc(100);
    step(2, 0, "second small object: same chunk, no new Alloc");
    vmem::free(s1);
    step(2, 0, "free one small object: the chunk stays");
    vmem::free(s2);
    step(0, 0, "free the last small object: the chunk goes back");
}

// limits 16 MiB device and 8 MiB host: fill both, hit the limits, move chunks around
void multi() {
    std::vector<void *> d;
    for (int i = 1; i <= 4; i++) {
        d.push_back(chunk());
        step(4 * i, 0, "alloc fills the device");
    }
    t::put(d[0], 4 * MiB, 9);
    void *h1 = chunk();
    step(16, 4, "alloc past the device limit: host");
    void *h2 = chunk();
    step(16, 8, "alloc past the device limit: host");
    t::expect(vmem::get().where(h1) == Loc::Host && vmem::get().where(h2) == Loc::Host, "where() says host for both");
    t::expect(chunk() == nullptr, "both limits full: nullptr");
    step(16, 8, "refused alloc changes nothing");
    t::expect(vmem::evict(d[0]) == Status::NoMemory, "evict with a full host: NoMemory");
    // B3: a refused move leaves the chunk where it was, with its data
    t::expect(vmem::get().where(d[0]) == Loc::Device && t::intact(d[0], 4 * MiB, 9), "refused evict: chunk still on device, data intact");
    step(16, 8, "refused evict changes nothing");
    vmem::free(h1);
    step(16, 4, "free a host chunk");
    t::expect(vmem::evict(d[0]) == Status::Ok, "evict after the free: Ok");
    t::expect(vmem::get().where(d[0]) == Loc::Host && t::intact(d[0], 4 * MiB, 9), "evicted chunk on host, data intact");
    step(12, 8, "evict moves 4 MiB to host");
    void *n = chunk();
    step(16, 8, "the freed device room takes a new chunk");
    t::expect(vmem::prefetch(d[0]) == Status::NoMemory, "prefetch with a full device: NoMemory");
    t::expect(vmem::get().where(d[0]) == Loc::Host && t::intact(d[0], 4 * MiB, 9), "refused prefetch: chunk still on host, data intact");
    step(16, 8, "refused prefetch changes nothing");
    vmem::free(n);
    step(12, 8, "free the new chunk");
    t::expect(vmem::prefetch(d[0]) == Status::Ok, "prefetch after the free: Ok");
    t::expect(vmem::get().where(d[0]) == Loc::Device && t::intact(d[0], 4 * MiB, 9), "prefetched chunk on device, data intact");
    step(16, 4, "prefetch moves 4 MiB back to device");
    vmem::free(h2);
    step(16, 0, "free the last host chunk");
    for (void *p : d) vmem::free(p);
    step(0, 0, "free everything");
}

// limits 20 MiB device (4 of them context overhead) and 8 MiB host. the checker restarts the broker at WAIT.
// while the link is broken the client creates, frees and moves chunks on its own; reconnect must report exactly the
// chunks alive then, at their current location. the checker also counts the Resync lines (3)
void resync() {
    void *a = chunk();
    void *b = chunk();
    step(8, 0, "two chunks on device");
    wait_for("restart");
    void *c = chunk();   // the old broker is gone: this send breaks the link and the chunk goes to the device locally
    t::expect(vmem::get().info().link == Link::Broken, "link is broken after the broker died");
    step(12, 0, "local mode: third chunk on device");
    vmem::free(chunk());   // made and freed while broken: must not be resynced
    t::expect(vmem::evict(a) == Status::Ok && vmem::get().where(a) == Loc::Host, "local mode: evict works without a broker");
    step(8, 4, "local mode: a on host");
    t::expect(vmem::get().reconnect() == Status::Ok, "reconnect: Ok");
    t::expect(vmem::get().info().link == Link::Connected, "link is connected again");
    step(8, 4, "counts unchanged by the reconnect");
    // the next replies depend on the resynced counts: had the broker missed c, or counted the freed chunk, these differ
    t::expect(vmem::prefetch(a) == Status::Ok, "prefetch after resync: Ok (device 4 + 12 + 4 <= 20)");
    step(12, 0, "a back on device");
    void *d = chunk();
    step(16, 0, "new broker counts the resynced chunks: the fourth fits");
    void *e = chunk();
    step(16, 4, "and the fifth goes to host");
    for (void *p : {a, b, c, d, e}) vmem::free(p);
    step(0, 0, "free everything");
}

// no device limit in the broker: the device really fills up, create fails and the chunk goes to host
void allocfail() {
    std::vector<void *> v;
    do v.push_back(vmem::alloc(256 * MiB));
    while (v.back() && vmem::get().used(Loc::Host) == 0 && v.size() < 64);
    t::expect(v.back() && vmem::get().used(Loc::Host) == 256 * MiB, "the chunk past VRAM landed on host");
    t::put(v.back(), 256 * MiB, 5);
    t::expect(t::intact(v.back(), 256 * MiB, 5), "the host chunk holds data");
    show();
    for (void *p : v) vmem::free(p);
    step(0, 0, "free everything");
}

// tenants: this one holds 8 MiB of the 16 MiB device while the next client runs
void hold() {
    void *a = chunk();
    void *b = chunk();
    step(8, 0, "two chunks on device");
    wait_for("other client");
    vmem::free(a);
    vmem::free(b);
    step(0, 0, "free everything");
}

// tenants: runs while hold has 8 MiB, so only 8 MiB of the device is left. the numbers after the view are the whole machine
void fill() {
    void *a = chunk();
    step(4, 0, "first chunk: device", 12, 0);
    void *b = chunk();
    step(8, 0, "second chunk: device is full", 16, 0);
    void *c = chunk();
    step(8, 4, "third chunk: host", 16, 4);
    void *d = chunk();
    step(8, 8, "fourth chunk: host", 16, 8);
    t::expect(chunk() == nullptr, "both limits full: nullptr");
    show(16, 8);
    for (void *p : {a, b, c, d}) vmem::free(p);
    step(0, 0, "free everything", 8, 0);
}

// the checker stops the broker with SIGSTOP: the next Alloc gets no reply, the receive timeout breaks the link and
// the chunk is made locally (B4). after SIGCONT the client reconnects and the broker counts both chunks
void hang() {
    void *a = chunk();
    step(4, 0, "one chunk on device");
    wait_for("stop");
    auto t0 = std::chrono::steady_clock::now();
    void *b = chunk();
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  alloc against the stopped broker returned after %.2f s\n", dt);
    // at least 1 s: the call really waited on the broker; under 5 s: the timeout fired
    t::expect(b && dt > 1 && dt < 5, "hung broker: alloc returns after the receive timeout, with a chunk");
    t::expect(vmem::get().info().link == Link::Broken, "hung broker: link is broken");
    step(8, 0, "local mode: second chunk on device");
    wait_for("resume");
    t::expect(vmem::get().reconnect() == Status::Ok, "reconnect after the broker resumed: Ok");
    step(8, 0, "the broker counts both chunks");
    vmem::free(a);
    vmem::free(b);
    step(0, 0, "free everything");
}

// no broker listens on the socket (missing file or stale file): init succeeds in local mode
void nobroker() {
    void *a = chunk();
    step(4, 0, "local mode: alloc works");
    t::expect(vmem::evict(a) == Status::Ok && vmem::get().where(a) == Loc::Host, "local mode: evict works");
    t::expect(vmem::get().reconnect() == Status::Disconnected, "reconnect with no broker: Disconnected");
    t::expect(vmem::get().info().link == Link::Broken, "after a failed reconnect the link is broken");
    step(0, 4, "failed reconnect changes nothing");
    vmem::free(a);
    step(0, 0, "free everything");
}

// limits 16/8 MiB. the checker restarts the broker at WAIT. the call that finds the broker dead falls back to local mode
// as usual; the next create reconnects by itself (auto_reconnect) and resyncs, so the new broker counts all three chunks
void autoreconnect() {
    void *a = chunk();
    void *b = chunk();
    step(8, 0, "two chunks on device");
    wait_for("restart");
    void *c = chunk();   // the send finds the old broker gone: link breaks, the chunk is made locally
    t::expect(vmem::get().info().link == Link::Broken, "link is broken after the broker died");
    step(12, 0, "local mode: third chunk on device");
    void *d = chunk();   // no reconnect() call: create tries the broker again
    t::expect(vmem::get().info().link == Link::Connected, "next create reconnected by itself");
    step(16, 0, "new broker counts the resynced chunks and the fourth");
    void *e = chunk();
    step(16, 4, "the fifth goes to host: the broker's counts are right");
    for (void *p : {a, b, c, d, e}) vmem::free(p);
    step(0, 0, "free everything");
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc != 3) {
        fprintf(stderr, "usage: state_client SCENARIO SOCKET (run it through tests/daemon/state_check.py)\n");
        return 2;
    }
    std::string s = argv[1];
    unsetenv("VMEM_BACKEND");
    unsetenv("VMEM_DAEMON");
    try {
        vmem::init({.backend = vmem::Kind::Vmem, .daemon = argv[2], .name = ("state-" + s).c_str(),
                    .auto_reconnect = s == "autoreconnect"});
        bool none = s == "nobroker";
        t::expect(vmem::get().info().link == (none ? Link::Local : Link::Connected),
                  none ? "no broker at init: link is local" : "link to vmemd is connected");
        if (s == "simple") simple();
        else if (s == "multi") multi();
        else if (s == "resync") resync();
        else if (s == "allocfail") allocfail();
        else if (s == "hold") hold();
        else if (s == "fill") fill();
        else if (s == "hang") hang();
        else if (s == "nobroker") nobroker();
        else if (s == "autoreconnect") autoreconnect();
        else {
            fprintf(stderr, "unknown scenario %s\n", s.c_str());
            return 2;
        }
        vmem::shutdown();
    } catch (const vmem::Error &e) {
        printf("FAIL: uncaught %s (%s)\n", e.what(), vmem::status_name(e.status));
        return 2;
    }
    return t::fails != 0;
}
