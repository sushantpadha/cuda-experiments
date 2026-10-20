// test 4: vmemd accounting, no GPU memory involved: make test N=4
// starts bin/vmemd with small limits, talks to it as fake tenants over the socket, checks every reply
// - config: mandatory file, unknown keys and bad values stop the daemon
// - scenarios: alloc, alloc_fail, move, move_done, resync, free, disconnect
// - random mix of calls from 3 tenants against a model of the broker's rules, then an exact OOM check
#include "utils.cuh"
#include "protocol.h"
#include "daemon.cuh"

#include <map>
#include <random>
#include <string>
#include <vector>

namespace proto = vmem::proto;
using vmem::Loc;
using vmem::Status;

constexpr Loc D = Loc::Device, H = Loc::Host;
constexpr size_t MiB = 1 << 20;

// ---- fake tenant ----

struct Tenant {
    int fd;
    explicit Tenant(const char *name) {
        fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        sockaddr_un a = {};
        a.sun_family = AF_UNIX;
        strcpy(a.sun_path, SOCK);
        if (connect(fd, (sockaddr *)&a, sizeof a)) throw vmem::Error(Status::Disconnected, "connect");
        proto::Msg m = {proto::Type::Hello};
        strncpy(m.name, name, proto::NAME_LEN - 1);
        ask(m);
    }
    ~Tenant() { close(fd); }

    void post(proto::Type t, uint64_t ptr, size_t size = 0, Loc l = D) {
        proto::Msg m = {t, l, ptr, size};
        send(fd, &m, sizeof m, MSG_NOSIGNAL);
    }
    proto::Reply ask(const proto::Msg &m) {
        proto::Reply r = {Status::Fatal};
        send(fd, &m, sizeof m, MSG_NOSIGNAL);
        if (recv(fd, &r, sizeof r, 0) != sizeof r) r.status = Status::Disconnected;
        return r;
    }

    proto::Reply alloc(uint64_t ptr, size_t mib) { return ask({proto::Type::Alloc, D, ptr, mib * MiB}); }
    Status move(uint64_t ptr, Loc to) { return ask({proto::Type::Move, to, ptr}).status; }
    void alloc_fail(uint64_t ptr) { post(proto::Type::AllocFail, ptr); }
    void move_done(uint64_t ptr, Loc at) { post(proto::Type::MoveDone, ptr, 0, at); }
    void resync(uint64_t ptr, size_t mib, Loc at) { post(proto::Type::Resync, ptr, mib * MiB, at); }
    void free(uint64_t ptr) { post(proto::Type::Free, ptr); }

    // messages without a reply are ordered only on this socket; another tenant's request can reach the broker
    // first. a Hello round trip makes sure everything sent before it has been handled
    void sync() {
        proto::Msg m = {proto::Type::Hello};
        ask(m);
    }
};

bool is(proto::Reply r, Status s, Loc l = D) { return r.status == s && (s != Status::Ok || r.loc == l); }

// a closed socket is handled on the broker's next loop turn, not in order with another tenant's messages
// ponytail: fixed sleep, a stats message would let the test wait for the exact state
void settle() { usleep(50000); }

// ---- tests ----

void config_tests() {
    printf("config: the file is mandatory, bad lines stop the daemon\n");
    int code = 0;
    t::expect(start("", "/tmp/vmemd-test4-missing.toml", &code) == -1 && code == 1, "missing config file: exit 1");
    t::expect(start("vram_limit_mib = 4\nbogus = 1\n", CONF, &code) == -1 && code == 1, "unknown key: exit 1");
    t::expect(start("vram_limit_mib = -2\n", CONF, &code) == -1 && code == 1, "value below -1: exit 1");
    t::expect(start("vram_limit_mib = 4 MiB\n", CONF, &code) == -1 && code == 1, "trailing text: exit 1");

    pid_t pid = start("# no limits\n\n");
    t::expect(pid > 0, "empty config starts, both limits unlimited");
    if (pid > 0) {
        Tenant a("a");
        t::expect(is(a.alloc(0x1000, 1 << 20), Status::Ok, D), "unlimited: 1 TiB goes to device");
        stop(pid);
    }
    pid = start("vram_limit_mib = -1\ndram_limit_mib = 0\n");
    if (pid > 0) {
        Tenant a("a");
        t::expect(is(a.alloc(0x1000, 1 << 20), Status::Ok, D), "-1: unlimited");
        stop(pid);
    }

    printf("context overhead: counts on the device from the first Hello on\n");
    pid = start("vram_limit_mib = 10\ndram_limit_mib = 0\nctx_overhead_mib = 4\n");
    t::expect(pid > 0, "ctx_overhead_mib is a known key");
    if (pid > 0) {
        {
            Tenant a("a");
            a.sync();   // a second Hello must not add it again
            t::expect(is(a.alloc(0x1000, 6), Status::Ok, D), "6 MiB fits next to one overhead (10/10)");
            t::expect(is(a.alloc(0x2000, 1), Status::NoMemory), "then the device is full");
        }
        settle();
        Tenant b("b"), c("c");
        t::expect(is(b.alloc(0x1000, 2), Status::Ok, D), "a left with its overhead; b and c hold 8, 2 MiB fits");
        t::expect(is(c.alloc(0x2000, 1), Status::NoMemory), "then the device is full");
        stop(pid);
    }
}

// limits: 6 MiB VRAM, 4 MiB DRAM
void scenario_tests() {
    pid_t pid = start("vram_limit_mib = 6\ndram_limit_mib = 4\n");
    if (pid < 0) return t::expect(false, "daemon starts with 6/4 MiB limits");

    printf("alloc: device until the VRAM limit, then host, then NoMemory\n");
    {
        Tenant a("a");
        t::expect(is(a.alloc(0x1000, 4), Status::Ok, D), "4 MiB: device (4/6)");
        t::expect(is(a.alloc(0x2000, 4), Status::Ok, H), "4 MiB: host, device too full (host 4/4)");
        t::expect(is(a.alloc(0x3000, 2), Status::Ok, D), "2 MiB: device (6/6)");
        t::expect(is(a.alloc(0x4000, 2), Status::NoMemory), "2 MiB: NoMemory");
        t::expect(is(a.alloc(0x1000, 2), Status::Unsupported), "same pointer twice: Unsupported");

        printf("alloc_fail: the chunk's bytes move from device to host\n");
        a.alloc_fail(0x3000);   // device 4/6, host 6/4 (over, accepted)
        a.alloc_fail(0x9999);   // unknown: ignored
        a.alloc_fail(0x2000);   // already on host: ignored
        t::expect(is(a.alloc(0x4000, 2), Status::Ok, D), "device room freed: 2 MiB on device (6/6)");

        printf("free: bytes come back at the chunk's location\n");
        a.free(0x4000);         // device 4/6
        a.free(0x9999);         // unknown: ignored
        t::expect(is(a.alloc(0x4000, 2), Status::Ok, D), "freed device room reused (6/6)");
        a.free(0x4000);
        a.free(0x4000);         // double free: ignored, counts stay
        t::expect(is(a.alloc(0x5000, 2), Status::Ok, D), "double free changed nothing (6/6)");
        t::expect(is(a.alloc(0x6000, 1), Status::NoMemory), "device and host full: NoMemory");
        a.free(0x2000);         // host 2/4
        a.free(0x3000);         // host 0/4
        t::expect(is(a.alloc(0x6000, 4), Status::Ok, H), "freed host room reused (4/4)");
    }
    settle();

    printf("disconnect: a tenant's usage goes away with its socket\n");
    {
        Tenant b("b");
        t::expect(is(b.alloc(0x1000, 6), Status::Ok, D), "6 MiB on device after a left");
        t::expect(is(b.alloc(0x2000, 4), Status::Ok, H), "4 MiB on host after a left");

        printf("move: reserves the target until move_done\n");
        b.free(0x2000);         // host 0/4
        t::expect(b.move(0x1000, H) == Status::NoMemory, "6 MiB to host: no room (limit 4)");
        b.free(0x1000);
        t::expect(is(b.alloc(0x1000, 4), Status::Ok, D), "4 MiB on device (4/6)");
        t::expect(is(b.alloc(0x2000, 2), Status::Ok, D), "2 MiB on device (6/6)");
        t::expect(b.move(0x1000, H) == Status::Ok, "4 MiB to host: granted (host 4/4 reserved)");
        t::expect(b.move(0x2000, H) == Status::NoMemory, "second move while one is open: refused");
        t::expect(b.move(0x2000, D) == Status::Ok, "move to where the chunk already is: Ok");
        t::expect(is(b.alloc(0x3000, 1), Status::NoMemory), "alloc during the move: host is reserved, NoMemory");
        b.move_done(0x1000, H); // device 2/6, host 4/4
        t::expect(is(b.alloc(0x3000, 2), Status::Ok, D), "after move_done: device room (4/6)");
        t::expect(b.move(0x1000, D) == Status::NoMemory, "4 MiB back to device: no room (4+4 > 6)");
        t::expect(b.move(0x9999, H) == Status::NoMemory, "move of an unknown chunk: refused");

        printf("move: a failed move and a freed moving chunk release the reservation\n");
        b.free(0x3000);         // device 2/6
        t::expect(b.move(0x1000, D) == Status::Ok, "4 MiB to device: granted (6/6 reserved)");
        b.move_done(0x1000, H); // failed: stays on host
        t::expect(is(b.alloc(0x3000, 4), Status::Ok, D), "failed move released the device room (6/6)");
        b.free(0x3000);
        t::expect(b.move(0x1000, D) == Status::Ok, "granted again");
        b.free(0x1000);         // freed while moving: drops the move and the chunk
        t::expect(is(b.alloc(0x4000, 4), Status::Ok, D), "device room back after free during a move (6/6)");
        t::expect(is(b.alloc(0x5000, 4), Status::Ok, H), "host room back too (4/4)");
    }
    settle();

    printf("resync: the broker trusts the report, even over a limit\n");
    {
        Tenant c("c");
        c.resync(0x1000, 8, D); // device 8/6
        t::expect(is(c.alloc(0x2000, 2), Status::Ok, H), "device over the limit: next chunk on host");
        c.resync(0x1000, 8, D); // already tracked: overwritten, not counted twice
        c.free(0x1000);         // device 0/6
        t::expect(is(c.alloc(0x3000, 6), Status::Ok, D), "after free the device is empty (6/6)");
        c.move_done(0x2000, D); // no grant: trusted, device 8/6, host 0/4
        t::expect(is(c.alloc(0x4000, 4), Status::Ok, H), "move_done without a grant moved the bytes (host 4/4)");
    }
    stop(pid);
}

// random calls from 3 tenants, each reply checked against a model; then everything is freed and the
// limits must be met exactly: 16 MiB on device, 8 MiB on host, then NoMemory
void random_tests() {
    printf("random: 3 tenants, 3000 calls, each reply checked against a model\n");
    const size_t VRAM = 16, DRAM = 8;
    pid_t pid = start("vram_limit_mib = 16\ndram_limit_mib = 8\n");
    if (pid < 0) return t::expect(false, "daemon starts with 16/8 MiB limits");

    struct Model {
        std::map<uint64_t, std::pair<size_t, Loc>> chunks;
    };
    std::vector<Tenant *> ts = {new Tenant("r0"), new Tenant("r1"), new Tenant("r2")};
    std::vector<Model> ms(ts.size());
    size_t used[2] = {};   // MiB, indexed by Loc
    auto at = [&](Loc l) -> size_t & { return used[(int)l]; };
    auto limit = [&](Loc l) { return l == D ? VRAM : DRAM; };

    std::mt19937 rng(42);
    uint64_t next = 0x1000;
    int bad = 0, ooms = 0;
    for (int i = 0; i < 3000; i++) {
        int k = rng() % ts.size();
        Tenant &tn = *ts[k];
        Model &m = ms[k];
        int op = rng() % 10;
        if (op < 4 || m.chunks.empty()) {   // alloc
            size_t mib = 1 + rng() % 3;
            Loc want = at(D) + mib <= VRAM ? D : H;
            bool fits = at(want) + mib <= limit(want);
            proto::Reply r = tn.alloc(next, mib);
            if (fits) {
                bad += !is(r, Status::Ok, want);
                m.chunks[next] = {mib, want};
                at(want) += mib;
            } else {
                bad += !is(r, Status::NoMemory);
                ooms++;
            }
            next += 0x1000;
            continue;
        }
        auto it = m.chunks.begin();
        std::advance(it, rng() % m.chunks.size());
        auto &[ptr, c] = *it;
        auto &[mib, loc] = c;
        if (op < 7) {   // free
            tn.free(ptr);
            tn.sync();
            at(loc) -= mib;
            m.chunks.erase(it);
        } else if (op < 9) {   // move, then move_done; one in four fails
            Loc to = loc == D ? H : D;
            bool fits = at(to) + mib <= limit(to);
            Status s = tn.move(ptr, to);
            bad += s != (fits ? Status::Ok : Status::NoMemory);
            if (s == Status::Ok) {
                Loc end = rng() % 4 ? to : loc;
                tn.move_done(ptr, end);
                tn.sync();
                at(loc) -= mib;
                at(end) += mib;
                loc = end;
            }
        } else if (loc == D) {   // alloc_fail; may push the host over its limit, as the broker allows
            tn.alloc_fail(ptr);
            tn.sync();
            at(D) -= mib;
            at(H) += mib;
            loc = H;
        }
    }
    printf("  %d of 3000 replies differ from the model, %d expected NoMemory\n", bad, ooms);
    t::expect(bad == 0, "every reply matches the model");
    t::expect(ooms > 0, "the run hit out of memory at least once");

    for (size_t k = 0; k < ts.size(); k++) {
        for (auto &[ptr, c] : ms[k].chunks) ts[k]->free(ptr);
        ts[k]->sync();
    }
    Tenant &a = *ts[0];
    bool exact = true;
    for (size_t i = 0; i < VRAM; i++) exact &= is(a.alloc(0x100000000 + i, 1), Status::Ok, D);
    for (size_t i = 0; i < DRAM; i++) exact &= is(a.alloc(0x200000000 + i, 1), Status::Ok, H);
    t::expect(exact, "after freeing everything: exactly 16 x 1 MiB on device, then 8 on host");
    t::expect(is(a.alloc(0x300000000, 1), Status::NoMemory), "then NoMemory");
    for (Tenant *tn : ts) delete tn;
    stop(pid);
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (access("bin/vmemd", X_OK)) {
        printf("FAIL: bin/vmemd missing (make daemon)\n");
        return 2;
    }
    unlink(LOG);
    config_tests();
    scenario_tests();
    random_tests();
    unlink(CONF);
    printf("daemon log: %s\n", LOG);
    printf(t::fails ? "FAIL: %d check(s)\n" : "PASS\n", t::fails);
    return t::fails != 0;
}
