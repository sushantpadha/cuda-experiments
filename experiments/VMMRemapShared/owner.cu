// Owner of a shared VMM segment. Creates the physical chunks, hands their fds to
// subscribers over a UNIX socket, and coordinates two things the subscribers
// take part in:
//   - growth: one extra chunk mid-run, streamed to everyone
//   - remap:  migrate a chunk device<->host via the cudaremap primitive, with a
//             revoke/remap handshake so subscribers drop and re-take the mapping
//
//   ./owner [n_subscribers] [n_chunks] [chunk_MB]

#include "common.cuh"
#include "remap.cuh"
#include "ipc_common.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

static CUdeviceptr g_base;
static size_t      g_chunk;
static std::vector<CUmemGenericAllocationHandle> g_handles;
static CUmemAccessDesc g_acc;

static void init_driver() {
    CU_CHECK( cuInit(0) );
    CUdevice d; CU_CHECK( cuDeviceGet(&d, 0) );
    CUcontext c; CU_CHECK( cuDevicePrimaryCtxRetain(&c, d) );
    CU_CHECK( cuCtxSetCurrent(c) );
}

// device-backed chunk, mapped at slot `idx`, returned as an exportable fd
static int add_chunk(size_t idx) {
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = 0;
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    CUmemGenericAllocationHandle h;
    CU_CHECK( cuMemCreate(&h, g_chunk, &p, 0) );
    CUdeviceptr at = g_base + idx * g_chunk;
    CU_CHECK( cuMemMap(at, g_chunk, 0, h, 0) );
    CU_CHECK( cuMemSetAccess(at, g_chunk, &g_acc, 1) );
    g_handles.push_back(h);

    int fd = -1;
    CU_CHECK( cuMemExportToShareableHandle(&fd, h,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) );
    return fd;
}

static void barrier(std::vector<int>& cs, const char* what) {
    for (int c : cs) { char b; if (read(c, &b, 1) != 1) die(what); }
}
static void broadcast(std::vector<int>& cs, Msg m, int fd = -1) {
    for (int c : cs)
        if (send_fds(c, &m, sizeof(m), fd >= 0 ? &fd : nullptr, fd >= 0 ? 1 : 0) < 0)
            die("broadcast");
}

// migrate chunk `idx` between device and host with subscribers attached:
//   revoke -> (subscribers unmap) -> remap backing under owner VA -> re-share
//   -> (subscribers re-import) -> done. Region contents survive.
static void remap_phase(std::vector<int>& cs, SharedState* st, size_t idx, bool to_host) {
    broadcast(cs, { MSG_REVOKE, g_chunk, 0, idx, 0 });
    barrier(cs, "revoke ack");

    remap_backing(g_base + idx * g_chunk, g_chunk, g_handles[idx], to_host, g_acc,
                  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    pthread_mutex_lock(&st->lock);
    st->chunk_on_host[idx] = to_host;
    pthread_mutex_unlock(&st->lock);

    int fd = -1;
    CU_CHECK( cuMemExportToShareableHandle(&fd, g_handles[idx],
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) );
    broadcast(cs, { MSG_REMAP, g_chunk, 0, idx, 1 }, fd);
    close(fd);
    barrier(cs, "remap ack");

    printf("[owner] chunk %zu migrated -> %s, all subscribers remapped\n",
           idx, to_host ? "HOST" : "DEVICE");
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int    nsub = argc > 1 ? atoi(argv[1]) : 2;
    size_t nchk = argc > 2 ? strtoull(argv[2], nullptr, 10) : 4;
    g_chunk     = (argc > 3 ? strtoull(argv[3], nullptr, 10) : 2) << 20;

    init_driver();
    g_chunk = ROUND_UP(g_chunk, remap_granularity());  // fits device AND host

    g_acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    g_acc.location.id = 0;
    g_acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CU_CHECK( cuMemAddressReserve(&g_base, g_chunk * MAX_CHUNKS, 0, 0, 0) );

    // ---- shared metadata: bump allocator + chunk-location table in POSIX shm ----
    shm_unlink(SHM_NAME);
    int sfd = shm_open(SHM_NAME, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (sfd < 0) die("shm_open");
    if (ftruncate(sfd, sizeof(SharedState)) < 0) die("ftruncate");
    auto* st = (SharedState*)mmap(nullptr, sizeof(SharedState),
                                  PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
    if (st == MAP_FAILED) die("mmap");
    memset(st, 0, sizeof(*st));
    pthread_mutexattr_t ma;
    pthread_mutexattr_init(&ma);
    pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&st->lock, &ma);
    st->chunk_size = g_chunk;
    st->chunk_count = nchk;
    st->total_bytes = g_chunk * nchk;

    // ---- initial chunks ----
    std::vector<int> fds;
    for (size_t i = 0; i < nchk; ++i) fds.push_back(add_chunk(i));
    printf("[owner] segment: %zu chunks x %zu MB, base=%p\n",
           nchk, g_chunk >> 20, (void*)g_base);

    // ---- UNIX socket listener ----
    unlink(SOCK_PATH);
    int lsock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (lsock < 0) die("socket");
    struct sockaddr_un a = {}; a.sun_family = AF_UNIX;
    strncpy(a.sun_path, SOCK_PATH, sizeof(a.sun_path) - 1);
    if (bind(lsock, (sockaddr*)&a, sizeof(a)) < 0) die("bind");
    if (listen(lsock, nsub) < 0) die("listen");
    printf("[owner] listening on %s for %d subscribers\n", SOCK_PATH, nsub);

    std::vector<int> cs(nsub);
    for (int i = 0; i < nsub; ++i) {
        cs[i] = accept(lsock, nullptr, nullptr);
        if (cs[i] < 0) die("accept");
        Msg m = { MSG_HEADER, g_chunk, (uint64_t)nchk, 0, (uint32_t)nchk };
        if (send_fds(cs[i], &m, sizeof(m), fds.data(), (int)nchk) < 0) die("send header");
        printf("[owner] subscriber %d connected, sent %zu fds\n", i, nchk);
    }

    barrier(cs, "read pass-1");
    printf("[owner] all subscribers did pass-1 alloc\n");

    // ---- grow the segment by one chunk ----
    int gfd = add_chunk(nchk);
    pthread_mutex_lock(&st->lock);
    st->chunk_count = nchk + 1;
    st->total_bytes = g_chunk * (nchk + 1);
    st->bump = g_chunk * nchk;   // pass-2 allocs land in the fresh chunk
    pthread_mutex_unlock(&st->lock);
    broadcast(cs, { MSG_ADD_CHUNK, g_chunk, (uint64_t)(nchk + 1), 0, 1 }, gfd);
    close(gfd);
    printf("[owner] grew segment to %zu chunks\n", nchk + 1);

    barrier(cs, "read pass-2");

    // ---- cudaremap, live: evict chunk 0 to host, then bring it back ----
    remap_phase(cs, st, 0, /*to_host=*/true);
    remap_phase(cs, st, 0, /*to_host=*/false);

    broadcast(cs, { MSG_GO, 0, 0, 0, 0 });

    // ---- owner verifies every region through its own mapping ----
    pthread_mutex_lock(&st->lock);
    int n = st->n_allocs;
    printf("[owner] verifying %d regions via owner base %p\n", n, (void*)g_base);
    for (int i = 0; i < n; ++i) {
        auto& al = st->allocs[i];
        uint32_t first = 0, last = 0;
        CU_CHECK( cuMemcpyDtoH(&first, g_base + al.off, 4) );
        CU_CHECK( cuMemcpyDtoH(&last,  g_base + al.off + al.len - 4, 4) );
        bool ok = (first == al.tag && last == al.tag);
        size_t chk = al.off / g_chunk;
        printf("  region off=%#lx len=%lu pid=%d tag=%#x chunk=%zu(%s)  read=%#x/%#x  %s\n",
               al.off, al.len, al.pid, al.tag, chk,
               st->chunk_on_host[chk] ? "host" : "dev", first, last,
               ok ? "OK" : "MISMATCH");
        if (!ok) { pthread_mutex_unlock(&st->lock); return 2; }
    }
    pthread_mutex_unlock(&st->lock);
    printf("=== owner: all %d cross-process regions verified after remap ===\n", n);

    for (int c : cs) close(c);
    close(lsock); unlink(SOCK_PATH);
    for (auto h : g_handles) cuMemRelease(h);
    cuMemUnmap(g_base, g_chunk * (nchk + 1));
    cuMemAddressFree(g_base, g_chunk * MAX_CHUNKS);
    munmap(st, sizeof(*st)); close(sfd); shm_unlink(SHM_NAME);
    return 0;
}
