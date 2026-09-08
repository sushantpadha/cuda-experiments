// Owner: creates a VMM segment, owns metadata, hands physical-memory fds to
// subscribers over a UNIX socket. Then acts as a barrier and verifies the
// shared view.
//   ./owner [n_subscribers] [n_chunks] [chunk_MB]

#include "common.cuh"
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

// create one device-backed chunk, map at slot `idx`, return an exportable fd
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
    CU_CHECK( cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) );
    return fd;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int    nsub  = argc > 1 ? atoi(argv[1]) : 2;
    size_t nchk  = argc > 2 ? strtoull(argv[2], nullptr, 10) : 4;
    g_chunk      = (argc > 3 ? strtoull(argv[3], nullptr, 10) : 2) << 20;

    init_driver();

    // align chunk to allocation granularity
    CUmemAllocationProp gp = {};
    gp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    gp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    size_t gran = 0;
    CU_CHECK( cuMemGetAllocationGranularity(&gran, &gp, CU_MEM_ALLOC_GRANULARITY_MINIMUM) );
    g_chunk = ROUND_UP(g_chunk, gran);

    g_acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    g_acc.location.id = 0;
    g_acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CU_CHECK( cuMemAddressReserve(&g_base, g_chunk * MAX_CHUNKS, 0, 0, 0) );

    // ---- shm metadata ----
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

    // ---- create initial chunks ----
    std::vector<int> fds;
    for (size_t i = 0; i < nchk; ++i) fds.push_back(add_chunk(i));
    printf("[owner] segment: %zu chunks x %zu MB, base=%p\n",
           nchk, g_chunk >> 20, (void*)g_base);

    // ---- UDS listener ----
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
        Msg m = { MSG_HEADER, (uint64_t)g_chunk, (uint64_t)nchk, (uint32_t)nchk };
        if (send_fds(cs[i], &m, sizeof(m), fds.data(), (int)nchk) < 0) die("send_fds");
        printf("[owner] subscriber %d connected, sent %zu fds\n", i, nchk);
    }

    // barrier 1: wait for each subscriber's first alloc
    for (int i = 0; i < nsub; ++i) { char b; if (read(cs[i], &b, 1) != 1) die("read D1"); }
    printf("[owner] all subscribers did pass-1 alloc\n");

    // ---- grow: one more chunk, streamed to every subscriber ----
    int gfd = add_chunk(nchk);
    pthread_mutex_lock(&st->lock);
    st->chunk_count = nchk + 1;
    st->total_bytes = g_chunk * (nchk + 1);
    pthread_mutex_unlock(&st->lock);
    for (int i = 0; i < nsub; ++i) {
        Msg m = { MSG_ADD_CHUNK, (uint64_t)g_chunk, (uint64_t)(nchk + 1), 1 };
        if (send_fds(cs[i], &m, sizeof(m), &gfd, 1) < 0) die("send add_chunk");
    }
    printf("[owner] grew segment to %zu chunks, streamed to subscribers\n", nchk + 1);

    // barrier 2: wait for each subscriber's second alloc (in the new chunk)
    for (int i = 0; i < nsub; ++i) { char b; if (read(cs[i], &b, 1) != 1) die("read D2"); }

    // release subscribers to verify
    for (int i = 0; i < nsub; ++i) {
        Msg m = { MSG_GO, 0, 0, 0 };
        if (send_fds(cs[i], &m, sizeof(m), nullptr, 0) < 0) die("send GO");
    }

    // ---- owner verifies the shared view through ITS OWN mapping ----
    pthread_mutex_lock(&st->lock);
    int n = st->n_allocs;
    printf("[owner] verifying %d regions via owner base %p\n", n, (void*)g_base);
    for (int i = 0; i < n; ++i) {
        auto& al = st->allocs[i];
        uint32_t w = 0;
        CU_CHECK( cuMemcpyDtoH(&w, g_base + al.off, sizeof(w)) );
        uint32_t last = 0;
        CU_CHECK( cuMemcpyDtoH(&last, g_base + al.off + al.len - 4, sizeof(last)) );
        bool ok = (w == al.tag && last == al.tag);
        printf("  region off=%#lx len=%lu pid=%d tag=%#x  read=%#x/%#x  %s\n",
               al.off, al.len, al.pid, al.tag, w, last, ok ? "OK" : "MISMATCH");
        if (!ok) { pthread_mutex_unlock(&st->lock); return 2; }
    }
    pthread_mutex_unlock(&st->lock);
    printf("=== owner: all %d cross-process regions verified ===\n", n);

    for (int i = 0; i < nsub; ++i) close(cs[i]);
    close(lsock); unlink(SOCK_PATH);
    for (auto h : g_handles) cuMemRelease(h);
    cuMemUnmap(g_base, g_chunk * (nchk + 1));
    cuMemAddressFree(g_base, g_chunk * MAX_CHUNKS);
    munmap(st, sizeof(*st)); close(sfd); shm_unlink(SHM_NAME);
    return 0;
}
