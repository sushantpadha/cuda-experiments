// vmemd: broker for libvmem tenants
// - one thread, one epoll loop over the listening socket and one socket per tenant
// - the broker does not own memory: it answers "device or host" and tracks usage
// - if tenant crashes or exits, the broker drops its accounted memory

// TODO: accounts, placement and the message handlers (see daemon/README.md)
// TODO: MPS support (tracker 2C)
//
//   bin/vmemd [--socket PATH] [--log PATH]      defaults: /tmp/vmemd.sock, /tmp/vmemd.log
//   tail -f /tmp/vmemd.log

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
    time_t connected;
    std::map<uint64_t, ChunkRec> chunks;   // chunk pointer -> size, location
    struct Move { uint64_t ptr; vmem::Loc to; };
    std::optional<Move> moving;            // granted, waiting for MoveDone; target bytes are reserved
    size_t device = 0, host = 0;           // bytes; device includes the context overhead
    size_t vram_limit = UNLIMITED, dram_limit = UNLIMITED;   // per-tenant quotas (1Bvi)
    int priority = 0;                      // unused until v2 (1Bviii)
};

// whole machine; the limits are what the broker hands out, below the capacity
struct Global {
    size_t vram_cap = 0, dram_cap = 0;
    size_t vram_limit = 0, dram_limit = 0;
    size_t ctx_overhead = 0;   // per tenant, added on Hello; not measured yet
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
    // TODO: subtract the tenant's device and host bytes from the global totals
    tenants.erase(fd);
    close(fd);   // also removes fd from epoll
}

// use MSG_NOSIGNAL: a dead tenant must not kill the broker with SIGPIPE
static void reply(int fd, vmem::Status s, vmem::Loc l = vmem::Loc::Device) {
    proto::Reply r{s, l};
    if (send(fd, &r, sizeof r, MSG_NOSIGNAL) != sizeof r)
        log("reply to fd %d failed: %s", fd, strerror(errno));   // recv will see the close next
}

// one message from one tenant; Alloc and Move expect a reply
static void handle(int fd, Tenant &t, const proto::Msg &m) {
    switch (m.type) {
    case proto::Type::Hello:
        t.name.assign(m.name, strnlen(m.name, proto::NAME_LEN));
        log("hello: pid %d is \"%s\"", t.pid, t.name.c_str());
        reply(fd, vmem::Status::Ok);
        break;
    // TODO: Alloc, AllocFail, Free, Move, MoveDone, Resync
    default:
        log("pid %d: unhandled message type %u", t.pid, (unsigned)m.type);
        if (m.type == proto::Type::Alloc || m.type == proto::Type::Move) reply(fd, vmem::Status::Unsupported);
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

// ---- main ----

int main(int argc, char **argv) {
    const char *path = proto::DEFAULT_SOCKET, *log_path = DEFAULT_LOG;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--socket") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) log_path = argv[++i];
        else {
            fprintf(stderr, "usage: vmemd [--socket PATH] [--log PATH]\n");
            return 2;
        }
    }

    // append, line-buffered, so tail -f sees each line as it is written
    if (!(logf = fopen(log_path, "a"))) {
        logf = stderr;
        die(log_path);
    }
    setvbuf(logf, nullptr, _IOLBF, 0);
    fprintf(stderr, "vmemd: logging to %s\n", log_path);

    int ls = listen_on(path);
    int ep = epoll_create1(0);
    if (ep < 0) die("epoll_create1");
    watch(ep, ls);
    log("listening on %s (pid %d)", path, getpid());

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
