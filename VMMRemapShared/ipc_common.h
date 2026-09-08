#pragma once
// Shared bits for the owner/subscriber multi-process demo:
//  - SharedState: allocator metadata in POSIX shm, one process-shared mutex
//  - UDS helpers: send/recv a message + a batch of file descriptors (SCM_RIGHTS)

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define SOCK_PATH   "/tmp/vmm_share.sock"
#define SHM_NAME    "/vmm_share_state"
#define MAX_CHUNKS  256
#define MAX_ALLOCS  64

// ponytail: bump allocator, no free; fixed-size tables; one global lock over the
// whole struct. Fine for a demo. Upgrade path: real free list + per-region locks.
struct SharedState {
    pthread_mutex_t lock;            // PTHREAD_PROCESS_SHARED, inited by owner
    uint64_t chunk_size;
    uint64_t chunk_count;
    uint64_t total_bytes;           // chunk_size * chunk_count
    uint64_t bump;                  // next free byte offset into the segment
    int      n_allocs;
    int      chunk_on_host[MAX_CHUNKS];   // 0 device, 1 host — observability only
    struct Alloc { uint64_t off; uint64_t len; int pid; uint32_t tag; } allocs[MAX_ALLOCS];
};

// wire messages owner -> subscriber
enum MsgType : uint32_t { MSG_HEADER = 1, MSG_ADD_CHUNK = 2, MSG_GO = 3 };
struct Msg {
    uint32_t type;
    uint64_t chunk_size;
    uint64_t chunk_count;   // MSG_HEADER: total now; MSG_ADD_CHUNK: new total
    uint32_t n_fds;         // fds carried alongside this message
};

// ---- SCM_RIGHTS fd passing -------------------------------------------------
inline int send_fds(int sock, const void* buf, size_t len, const int* fds, int nfds) {
    struct iovec io = { (void*)buf, len };
    char cbuf[CMSG_SPACE(sizeof(int) * 64)];
    memset(cbuf, 0, sizeof(cbuf));
    struct msghdr m = {};
    m.msg_iov = &io; m.msg_iovlen = 1;
    if (nfds > 0) {
        m.msg_control = cbuf;
        m.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);
        struct cmsghdr* c = CMSG_FIRSTHDR(&m);
        c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int) * nfds);
        memcpy(CMSG_DATA(c), fds, sizeof(int) * nfds);
    }
    return sendmsg(sock, &m, 0) < 0 ? -1 : 0;
}

inline int recv_fds(int sock, void* buf, size_t len, int* fds, int max_fds, int* got_fds) {
    struct iovec io = { buf, len };
    char cbuf[CMSG_SPACE(sizeof(int) * 64)];
    struct msghdr m = {};
    m.msg_iov = &io; m.msg_iovlen = 1;
    m.msg_control = cbuf; m.msg_controllen = sizeof(cbuf);
    ssize_t r = recvmsg(sock, &m, 0);
    if (r <= 0) return -1;
    *got_fds = 0;
    for (struct cmsghdr* c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
            int n = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            if (n > max_fds) n = max_fds;
            memcpy(fds, CMSG_DATA(c), sizeof(int) * n);
            *got_fds = n;
        }
    }
    return 0;
}

inline void die(const char* what) { perror(what); exit(1); }
