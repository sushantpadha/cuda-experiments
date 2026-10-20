// vmemd: broker for libvmem tenants
// - one thread, one epoll loop over the listening socket and one socket per tenant
// - the broker does not own memory: it answers "device or host" and tracks usage
// - if tenant crashes or exits, the broker drops its accounted memory

// TODO: MPS support (tracker 2C)
//
//   bin/vmemd --config daemon/vmemd.toml [--socket PATH] [--log PATH]      defaults: /tmp/vmemd.sock, /tmp/vmemd.log
//   tail -f /tmp/vmemd.log
// ! resync and allocfail are authoritative; the broker does not try to recover just tracks if any limits are overrun

#include "protocol.h"

#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace proto = vmem::proto;

// ---- state ----

constexpr const char *DEFAULT_LOG = "/tmp/vmemd.log";
constexpr size_t UNLIMITED = SIZE_MAX;

struct ChunkRec {
    size_t size;
    vmem::Loc loc;
};

struct Tenant {
    pid_t pid;
    std::string name;   // from Hello; empty until then
    bool hello = false; // the first Hello adds the context overhead; later ones only sync
    time_t connected;
    std::map<uint64_t, ChunkRec> chunks;   // chunk pointer -> size, location
    struct Move { uint64_t ptr; vmem::Loc to; };
    std::optional<Move> moving;            // granted, waiting for MoveDone; target bytes are reserved
    size_t device = 0, host = 0;           // bytes; device includes the context overhead
    size_t vram_limit = UNLIMITED, dram_limit = UNLIMITED;   // per-tenant quotas (1Bvi)
    int priority = 0;                      // unused until v2 (1Bviii)
};

// whole machine; the limits come from the config file
struct Global {
    size_t vram_limit = UNLIMITED, dram_limit = UNLIMITED;
    size_t ctx_overhead = 0;   // per tenant, added to its device usage on its first Hello
    size_t device = 0, host = 0;   // sums over tenants
};

static std::map<int, Tenant> tenants;   // socket fd -> tenant
static Global g;
static FILE *logf = stderr;             // the log file once main opens it

// ---- logging ----

static void log(const char *f, ...) __attribute__((format(printf, 1, 2)));
static void log(const char *f, ...) {
    char ts[32];
    time_t now = time(nullptr);
    strftime(ts, sizeof ts, "%H:%M:%S", localtime(&now));
    fprintf(logf, "[vmemd %s] ", ts);
    va_list ap;
    va_start(ap, f);
    vfprintf(logf, f, ap);
    va_end(ap);
    fprintf(logf, "\n");
}

static void die(const char *what) {
    log("%s: %s", what, strerror(errno));
    exit(1);
}

// ---- socket setup ----

static sockaddr_un address(const char *path) {
    sockaddr_un a = {};
    a.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof a.sun_path) {
        log("socket path too long (max %zu): %s", sizeof a.sun_path - 1, path);
        exit(1);
    }
    strcpy(a.sun_path, path);
    return a;
}

// bind the listening socket; refuses to start if another broker answers on the path
static int listen_on(const char *path) {
    sockaddr_un a = address(path);

    // a successful connect means a live broker owns the path; unlinking would steal it
    int probe = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (connect(probe, (sockaddr *)&a, sizeof a) == 0) {
        log("another vmemd is already listening on %s", path);
        exit(1);
    }
    close(probe);

    unlink(path);   // stale file left by a broker that died
    int fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (fd < 0) die("socket");
    if (bind(fd, (sockaddr *)&a, sizeof a)) die("bind");
    if (listen(fd, 64)) die("listen");
    return fd;
}

static void watch(int ep, int fd) {
    epoll_event ev = {};
    ev.events = EPOLLIN;
    ev.data.fd = fd;
    if (epoll_ctl(ep, EPOLL_CTL_ADD, fd, &ev)) die("epoll_ctl");
}

// ---- tenants ----

static void on_connect(int ep, int ls) {
    int fd = accept(ls, nullptr, nullptr);
    if (fd < 0) {
        log("accept: %s", strerror(errno));   // e.g. out of fds; the broker keeps running
        return;
    }
    ucred cr = {};
    socklen_t len = sizeof cr;
    // read the peer's credentials to get PID etc.
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cr, &len)) {
        log("SO_PEERCRED: %s", strerror(errno));
        close(fd);
        return;
    }
    watch(ep, fd);
    Tenant &t = tenants[fd];
    t.pid = cr.pid;
    t.connected = time(nullptr);
    log("connect: pid %d (fd %d)", cr.pid, fd);
}

// clean exit and crash look the same: the socket closed
static void on_disconnect(int fd) {
    Tenant &t = tenants.at(fd);
    log("disconnect: pid %d \"%s\", connected %lds", t.pid, t.name.c_str(), (long)(time(nullptr) - t.connected));
    g.device -= t.device;
    g.host -= t.host;
    tenants.erase(fd);
    close(fd);   // also removes fd from epoll
}

// use MSG_NOSIGNAL: a dead tenant must not kill the broker with SIGPIPE
static void reply(int fd, vmem::Status s, vmem::Loc l = vmem::Loc::Device) {
    proto::Reply r{s, l};
    if (send(fd, &r, sizeof r, MSG_NOSIGNAL) != sizeof r)
        log("reply to fd %d failed: %s", fd, strerror(errno));   // recv will see the close next
}

// ---- accounting ----

static size_t &used_bytes(Tenant &t, vmem::Loc l) { return l == vmem::Loc::Device ? t.device : t.host; }
static size_t &used_bytes(vmem::Loc l) { return l == vmem::Loc::Device ? g.device : g.host; }

// room for size more bytes at l, under both the tenant's and the global limit
static bool fits_limit(Tenant &t, vmem::Loc l, size_t size) {
    bool dev = l == vmem::Loc::Device;
    size_t tl = dev ? t.vram_limit : t.dram_limit, gl = dev ? g.vram_limit : g.dram_limit;
    return used_bytes(t, l) + size <= tl && used_bytes(l) + size <= gl;
}

// add or remove a chunk's bytes at l, for the tenant and the global total
static void update_count(Tenant &t, vmem::Loc l, size_t size, bool add) {
    if (add) used_bytes(t, l) += size, used_bytes(l) += size;
    else used_bytes(t, l) -= size, used_bytes(l) -= size;
}

// ! are we going over a limit after accepting a chunk the broker could not refuse (AllocFail, Resync)
static void check_over_limit(Tenant &t, vmem::Loc l) {
    bool dev = l == vmem::Loc::Device;
    size_t tl = dev ? t.vram_limit : t.dram_limit, gl = dev ? g.vram_limit : g.dram_limit;
    if (used_bytes(t, l) > tl || used_bytes(l) > gl)
        log("over limit on %s: pid %d holds %zu MiB, all tenants %zu MiB of %zu MiB", vmem::loc_name(l), t.pid,
            used_bytes(t, l) >> 20, used_bytes(l) >> 20, gl >> 20);
}

// ---- handlers ----

// device until the VRAM limit, then host until the DRAM limit, then NoMemory
static void handle_alloc(Tenant &t, size_t size, uint64_t ptr, vmem::Status *s, vmem::Loc *l) {
    // check that the chunk is not already tracked
    if (t.chunks.count(ptr)) {
        log("pid %d: alloc for %#lx, which it already holds; refused", t.pid, (unsigned long)ptr);
        *s = vmem::Status::Unsupported;
        return;
    }
    if (fits_limit(t, vmem::Loc::Device, size)) *l = vmem::Loc::Device;
    else if (fits_limit(t, vmem::Loc::Host, size)) *l = vmem::Loc::Host;
    else {
        log("pid %d: alloc of %zu MiB refused, no room on device or host", t.pid, size >> 20);
        *s = vmem::Status::NoMemory;
        return;
    }
    t.chunks[ptr] = {size, *l};
    update_count(t, *l, size, true);
    *s = vmem::Status::Ok;
    log("alloc: pid %d, %zu MiB %s in the counts", t.pid, size >> 20, vmem::loc_name(*l));
}

// the device cuMemCreate failed after the broker said device; the chunk is on host now
static void handle_alloc_fail(Tenant &t, uint64_t ptr) {
    // check that the chunk is tracked
    auto it = t.chunks.find(ptr);
    if (it == t.chunks.end()) {
        log("pid %d: alloc_fail for unknown chunk %#lx; ignored", t.pid, (unsigned long)ptr);
        return;
    }
    ChunkRec &c = it->second;
    if (c.loc != vmem::Loc::Device) {
        log("pid %d: alloc_fail for %#lx, which is already on host; ignored", t.pid, (unsigned long)ptr);
        return;
    }
    update_count(t, vmem::Loc::Device, c.size, false);  // subtract from device
    update_count(t, vmem::Loc::Host, c.size, true);     // add to host
    c.loc = vmem::Loc::Host;
    log("alloc_fail: pid %d, %zu MiB moved from device to host in the counts", t.pid, c.size >> 20);
    check_over_limit(t, vmem::Loc::Host);
}

// evict or prefetch: grant if the target has room, and reserve the bytes there until MoveDone
// ! expects a MoveDone to be sent
static vmem::Status handle_move(Tenant &t, uint64_t ptr, vmem::Loc to) {
    // check that the chunk is tracked
    auto it = t.chunks.find(ptr);
    if (it == t.chunks.end()) {
        log("pid %d: move for unknown chunk %#lx; refused", t.pid, (unsigned long)ptr);
        return vmem::Status::NoMemory;
    }
    ChunkRec &c = it->second;
    if (c.loc == to) return vmem::Status::Ok;   // nothing to move, nothing to reserve
    // check that no other chunk is moving; only move one at a time
    // TODO: allow moving >1 chunks in v2
    if (t.moving) {
        log("pid %d: move for %#lx while %#lx is still moving; refused", t.pid, (unsigned long)ptr,
            (unsigned long)t.moving->ptr);
        return vmem::Status::NoMemory;
    }
    if (!fits_limit(t, to, c.size)) {
        log("pid %d: move of %zu MiB to %s refused, no room", t.pid, c.size >> 20, vmem::loc_name(to));
        return vmem::Status::NoMemory;
    }
    update_count(t, to, c.size, true);   // reserve; the chunk still counts at its old location too
    t.moving = Tenant::Move{ptr, to};
    log("move: pid %d, %zu MiB %s -> %s granted", t.pid, c.size >> 20, vmem::loc_name(c.loc), vmem::loc_name(to));
    return vmem::Status::Ok;
}

// the move finished or failed
// ! can recieve Move Done without corresponding Move -- trust the tenant
static void handle_move_done(Tenant &t, uint64_t ptr, vmem::Loc at) {
    // check if the chunk is tracked
    auto it = t.chunks.find(ptr);
    if (it == t.chunks.end()) {
        log("pid %d: move_done for unknown chunk %#lx; ignored", t.pid, (unsigned long)ptr);
        return;
    }
    // check that the chunk is moving
    ChunkRec &c = it->second;
    if (!t.moving || t.moving->ptr != ptr) {
        // no grant to release; trust the tenant about where the chunk is
        log("pid %d: move_done for %#lx without a granted move; recording it on %s", t.pid, (unsigned long)ptr,
            vmem::loc_name(at));
    } else {
        update_count(t, t.moving->to, c.size, false);   // release the reservation
        if (at != t.moving->to) log("pid %d: move of %#lx failed, stays on %s", t.pid, (unsigned long)ptr, vmem::loc_name(at));
        t.moving.reset();
    }
    // final update counters
    if (at != c.loc) {
        update_count(t, c.loc, c.size, false);
        update_count(t, at, c.size, true);
        c.loc = at;
        check_over_limit(t, at);   // ! only possible without a granted Move
    }
}

// after a broker restart the tenant reports each live chunk; the broker trusts it, even over a limit
// ! tenant might report total usage > limit, trust it but still keep the same limit
static void handle_resync(Tenant &t, uint64_t ptr, size_t size, vmem::Loc at) {
    // check that the chunk is not already tracked
    auto it = t.chunks.find(ptr);
    if (it != t.chunks.end()) {
        log("pid %d: resync for %#lx, which is already tracked; overwriting", t.pid, (unsigned long)ptr);
        update_count(t, it->second.loc, it->second.size, false);
    }
    t.chunks[ptr] = {size, at};
    update_count(t, at, size, true);
    // possible in this case
    check_over_limit(t, at);
}


static void handle_free(Tenant &t, uint64_t ptr) {
    // check that the chunk is tracked
    auto it = t.chunks.find(ptr);
    if (it == t.chunks.end()) {
        log("pid %d: free for unknown chunk %#lx; ignored", t.pid, (unsigned long)ptr);
        return;
    }
    ChunkRec &c = it->second;
    // check that the chunk is not moving; if moving, drop its reserved counts (moving->to)
    if (t.moving && t.moving->ptr == ptr) {
        log("pid %d: free for %#lx while it is moving; dropping the move", t.pid, (unsigned long)ptr);
        update_count(t, t.moving->to, c.size, false);
        t.moving.reset();
    }
    update_count(t, c.loc, c.size, false);
    log("free: pid %d, %zu MiB %s", t.pid, c.size >> 20, vmem::loc_name(c.loc));
    t.chunks.erase(it);
}

// routing logic: one message from one tenant; Alloc and Move expect a reply
static void handle(int fd, Tenant &t, const proto::Msg &m) {
    switch (m.type) {
    case proto::Type::Hello:
        if (!t.hello) {
            t.hello = true;
            t.name.assign(m.name, strnlen(m.name, proto::NAME_LEN));
            update_count(t, vmem::Loc::Device, g.ctx_overhead, true);
            log("hello: pid %d is \"%s\"", t.pid, t.name.c_str());
        }
        reply(fd, vmem::Status::Ok);
        break;
    case proto::Type::Alloc: {
        vmem::Status s = vmem::Status::NoMemory;
        vmem::Loc l = vmem::Loc::Host;
        handle_alloc(t, m.size, m.ptr, &s, &l);
        reply(fd, s, l);
        break;
    }
    case proto::Type::AllocFail:
        handle_alloc_fail(t, m.ptr);
        break;
    case proto::Type::Move:
        reply(fd, handle_move(t, m.ptr, m.loc));
        break;
    case proto::Type::MoveDone:
        handle_move_done(t, m.ptr, m.loc);
        break;
    case proto::Type::Resync:
        handle_resync(t, m.ptr, m.size, m.loc);
        break;
    case proto::Type::Free:
        handle_free(t, m.ptr);
        break;
    default:
        log("pid %d: unhandled message type %u", t.pid, (unsigned)m.type);
    }
}

static void on_readable(int fd) {
    proto::Msg m;
    ssize_t n = recv(fd, &m, sizeof m, 0); // read exact size because its a SOCK_SEQPACKET
    if (n < 0 && errno == EINTR) return;   // level-triggered: epoll reports it again
    if (n <= 0) return on_disconnect(fd);
    if (n != sizeof m) {
        log("pid %d: %zd-byte message, expected %zu; dropping the tenant", tenants.at(fd).pid, n, sizeof m);
        return on_disconnect(fd);
    }
    handle(fd, tenants.at(fd), m);
}

// ---- config ----

// exits on a missing file, an unknown key or a bad value
static void read_config(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) die(path);
    char line[256];
    for (int n = 1; fgets(line, sizeof line, f); n++) {
        if (char *c = strchr(line, '#')) *c = 0;
        if (strspn(line, " \t\r\n") == strlen(line)) continue;   // blank or comment only
        char key[64], rest[8];
        long long v;
        if (sscanf(line, " %63[a-z_] = %lld %7s", key, &v, rest) != 2 || v < -1) {
            log("%s:%d: expected \"key = number\" with number >= -1", path, n);
            exit(1);
        }
        size_t bytes = v == -1 ? UNLIMITED : (size_t)v << 20;
        if (!strcmp(key, "vram_limit_mib")) g.vram_limit = bytes;
        else if (!strcmp(key, "dram_limit_mib")) g.dram_limit = bytes;
        else if (!strcmp(key, "ctx_overhead_mib")) g.ctx_overhead = v == -1 ? 0 : bytes;   // -1: none
        else {
            log("%s:%d: unknown key %s", path, n, key);
            exit(1);
        }
    }
    fclose(f);
}

static std::string limit_str(size_t l) { return l == UNLIMITED ? "unlimited" : std::to_string(l >> 20) + " MiB"; }

// ---- main ----

int main(int argc, char **argv) {
    const char *path = proto::DEFAULT_SOCKET, *log_path = DEFAULT_LOG, *config = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--socket") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) log_path = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else config = nullptr, i = argc;   // unknown argument: print usage
    }
    if (!config) {
        fprintf(stderr, "usage: vmemd --config PATH [--socket PATH] [--log PATH]\n");
        return 2;
    }

    // append, line-buffered, so tail -f sees each line as it is written
    if (!(logf = fopen(log_path, "a"))) {
        logf = stderr;
        die(log_path);
    }
    setvbuf(logf, nullptr, _IOLBF, 0);
    fprintf(stderr, "vmemd: logging to %s\n", log_path);
    read_config(config);

    int ls = listen_on(path);
    int ep = epoll_create1(0);
    if (ep < 0) die("epoll_create1");
    watch(ep, ls);
    log("listening on %s (pid %d), VRAM limit %s, DRAM limit %s, context overhead %zu MiB", path, getpid(),
        limit_str(g.vram_limit).c_str(), limit_str(g.dram_limit).c_str(), g.ctx_overhead >> 20);

    for (;;) {
        epoll_event evs[64];
        int n = epoll_wait(ep, evs, 64, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            die("epoll_wait");
        }
        for (int i = 0; i < n; i++) {
            int fd = evs[i].data.fd;
            if (fd == ls) on_connect(ep, ls);
            else on_readable(fd);
        }
    }
}
