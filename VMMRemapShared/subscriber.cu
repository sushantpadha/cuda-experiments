// A tenant. Attaches to the owner's VMM segment, maps the same physical chunks
// at its OWN virtual base, bump-allocates a region from the shared metadata,
// tags it on the GPU. Then it participates in the owner's remap handshake
// (unmap the revoked chunk, re-import it afterwards) and finally verifies every
// tenant's region through its own address space.
//
//   ./subscriber [label]

#include "common.cuh"
#include "ipc_common.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <ctime>

static CUdeviceptr g_base;
static size_t      g_chunk;
static CUmemAccessDesc g_acc;
static std::vector<CUmemGenericAllocationHandle> g_imported;  // per chunk

static void init_driver() {
    CU_CHECK( cuInit(0) );
    CUdevice d; CU_CHECK( cuDeviceGet(&d, 0) );
    CUcontext c; CU_CHECK( cuDevicePrimaryCtxRetain(&c, d) );
    CU_CHECK( cuCtxSetCurrent(c) );
}

// import an fd and map it at slot `idx` in our own address space
static void map_chunk(size_t idx, int fd) {
    CUmemGenericAllocationHandle h;
    CU_CHECK( cuMemImportFromShareableHandle(&h, (void*)(uintptr_t)fd,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) );
    CUdeviceptr at = g_base + idx * g_chunk;
    CU_CHECK( cuMemMap(at, g_chunk, 0, h, 0) );
    CU_CHECK( cuMemSetAccess(at, g_chunk, &g_acc, 1) );
    if (g_imported.size() <= idx) g_imported.resize(idx + 1);
    g_imported[idx] = h;
}

static void unmap_chunk(size_t idx) {
    CU_CHECK( cuMemUnmap(g_base + idx * g_chunk, g_chunk) );
    CU_CHECK( cuMemRelease(g_imported[idx]) );
}

// bump-allocate `len` bytes from shared state, tag on the GPU, record the entry
static void alloc_and_tag(SharedState* st, uint64_t len, uint32_t tag, int pid) {
    pthread_mutex_lock(&st->lock);                 // ponytail: one global lock
    uint64_t off = st->bump;
    if (off + len > st->total_bytes) { pthread_mutex_unlock(&st->lock); die("segment OOM"); }
    st->bump += len;
    int idx = st->n_allocs++;
    st->allocs[idx] = { off, len, pid, tag };
    pthread_mutex_unlock(&st->lock);

    CU_CHECK( cuMemsetD32(g_base + off, tag, len / 4) );
    CU_CHECK( cuCtxSynchronize() );
    printf("[sub %d] wrote region off=%#lx len=%lu tag=%#x via base %p\n",
           pid, off, len, tag, (void*)g_base);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* label = argc > 1 ? argv[1] : "sub";
    int pid = getpid();

    init_driver();

    // attach shm (owner creates it; brief retry)
    int sfd = -1;
    for (int i = 0; i < 400 && sfd < 0; ++i) {
        sfd = shm_open(SHM_NAME, O_RDWR, 0600);
        if (sfd < 0) { struct timespec ts = {0, 5'000'000}; nanosleep(&ts, nullptr); }
    }
    if (sfd < 0) die("shm_open");
    auto* st = (SharedState*)mmap(nullptr, sizeof(SharedState),
                                  PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
    if (st == MAP_FAILED) die("mmap");

    // connect to owner
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un a = {}; a.sun_family = AF_UNIX;
    strncpy(a.sun_path, SOCK_PATH, sizeof(a.sun_path) - 1);
    for (int i = 0; i < 400; ++i) {
        if (connect(sock, (sockaddr*)&a, sizeof(a)) == 0) break;
        struct timespec ts = {0, 5'000'000}; nanosleep(&ts, nullptr);
    }

    // HEADER + all current fds
    Msg m; int fds[MAX_CHUNKS]; int nfds = 0;
    if (recv_fds(sock, &m, sizeof(m), fds, MAX_CHUNKS, &nfds) < 0) die("recv header");
    g_chunk = m.chunk_size;

    g_acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    g_acc.location.id = 0;
    g_acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // reserve our base at a per-process offset so tenants get DIFFERENT bases
    // -> shared allocations must be offset-based, never pointer-based
    CUdeviceptr raw;
    size_t pad = g_chunk * ((pid % 5) + 1);
    CU_CHECK( cuMemAddressReserve(&raw, g_chunk * MAX_CHUNKS + pad, 0, 0, 0) );
    g_base = raw + pad;

    for (int i = 0; i < nfds; ++i) map_chunk(i, fds[i]);
    printf("[%s %d] mapped %d chunks at own base %p\n", label, pid, nfds, (void*)g_base);

    // pass 1
    alloc_and_tag(st, 64u << 10, 0xA5000000u | (pid & 0xFFFF), pid);
    if (write(sock, "D", 1) != 1) die("write pass-1 ack");

    // ADD_CHUNK
    if (recv_fds(sock, &m, sizeof(m), fds, MAX_CHUNKS, &nfds) < 0) die("recv add_chunk");
    if (m.type != MSG_ADD_CHUNK || nfds != 1) die("expected ADD_CHUNK");
    map_chunk(m.chunk_count - 1, fds[0]);
    printf("[%s %d] mapped grown chunk, now %lu chunks\n", label, pid, m.chunk_count);

    // pass 2 (into the grown chunk)
    alloc_and_tag(st, 64u << 10, 0x5A000000u | (pid & 0xFFFF), pid);
    if (write(sock, "D", 1) != 1) die("write pass-2 ack");

    // ---- remap handshake loop: serve REVOKE/REMAP until GO ----
    for (;;) {
        if (recv_fds(sock, &m, sizeof(m), fds, MAX_CHUNKS, &nfds) < 0) die("recv ctrl");
        if (m.type == MSG_GO) break;
        if (m.type == MSG_REVOKE) {
            unmap_chunk(m.chunk_idx);
            printf("[%s %d] revoked chunk %lu\n", label, pid, m.chunk_idx);
            if (write(sock, "D", 1) != 1) die("write revoke ack");
        } else if (m.type == MSG_REMAP) {
            map_chunk(m.chunk_idx, fds[0]);
            printf("[%s %d] re-mapped chunk %lu (%s)\n", label, pid, m.chunk_idx,
                   st->chunk_on_host[m.chunk_idx] ? "host" : "dev");
            if (write(sock, "D", 1) != 1) die("write remap ack");
        } else {
            die("unexpected ctrl msg");
        }
    }

    // ---- verify every tenant's region through our own VA ----
    pthread_mutex_lock(&st->lock);
    int n = st->n_allocs, ok = 0;
    for (int i = 0; i < n; ++i) {
        auto& al = st->allocs[i];
        uint32_t first = 0, last = 0;
        CU_CHECK( cuMemcpyDtoH(&first, g_base + al.off, 4) );
        CU_CHECK( cuMemcpyDtoH(&last,  g_base + al.off + al.len - 4, 4) );
        bool good = (first == al.tag && last == al.tag);
        printf("[%s %d] region %d off=%#lx pid=%d tag=%#x -> %#x/%#x %s\n",
               label, pid, i, al.off, al.pid, al.tag, first, last, good ? "OK" : "BAD");
        ok += good;
    }
    pthread_mutex_unlock(&st->lock);

    bool pass = (ok == n && n >= 4);   // 2 tenants x 2 passes
    printf("[%s %d] verified %d/%d regions via own mapping — %s\n",
           label, pid, ok, n, pass ? "PASS" : "FAIL");

    for (auto h : g_imported) cuMemRelease(h);
    close(sock); munmap(st, sizeof(*st)); close(sfd);
    return pass ? 0 : 1;
}
