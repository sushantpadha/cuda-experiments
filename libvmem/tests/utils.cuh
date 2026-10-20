// shared by the tests: fill/check pattern kernels, check printing, init/shutdown around a test body,
// starting bin/vmemd for the "d" runs, and the line protocol of the daemon test clients
#pragma once

#include "libvmem.cuh"

#include <cuda_runtime.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// fill / check a buffer with a known pattern: word i holds k * (i + 1), k != 0
// both block until the kernel is done
namespace pattern {

__device__ unsigned long long d_bad;

__global__ void fill_k(unsigned *p, size_t n, unsigned k) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = k * (unsigned)(i + 1);
}
// ponytail: one atomic per wrong word, slow if a whole large buffer is wrong; per-block counts if that matters
__global__ void check_k(const unsigned *p, size_t n, unsigned k) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != k * (unsigned)(i + 1)) atomicAdd(&d_bad, 1ull);
}

// n words; returns the first launch or kernel error
inline cudaError_t fill(unsigned *p, size_t n, unsigned k) {
    fill_k<<<160, 256>>>(p, n, k);
    cudaError_t e = cudaGetLastError();
    return e != cudaSuccess ? e : cudaDeviceSynchronize();
}

// n words; wrong-word count in bad
inline cudaError_t check(const unsigned *p, size_t n, unsigned k, unsigned long long &bad) {
    bad = 0;
    cudaError_t e = cudaMemcpyToSymbol(d_bad, &bad, sizeof bad);
    if (e != cudaSuccess) return e;
    check_k<<<160, 256>>>(p, n, k);
    if ((e = cudaGetLastError()) != cudaSuccess || (e = cudaDeviceSynchronize()) != cudaSuccess) return e;
    return cudaMemcpyFromSymbol(&bad, d_bad, sizeof bad);
}

}  // namespace pattern

namespace t {

inline int fails = 0;

inline void expect(bool ok, const char *what) {
    printf("  %s: %s\n", ok ? "ok" : "FAIL", what);
    fails += !ok;
}

// f must throw vmem::Error with this status
template <class F> void expect_throws(vmem::Status s, F f, const char *what) {
    bool ok = false;
    try { f(); } catch (const vmem::Error &e) { ok = e.status == s; }
    expect(ok, what);
}

inline void put(void *p, size_t bytes, unsigned k) { pattern::fill((unsigned *)p, bytes / 4, k); }

inline bool intact(void *p, size_t bytes, unsigned k) {
    unsigned long long bad = 1;
    return pattern::check((unsigned *)p, bytes / 4, k, bad) == cudaSuccess && bad == 0;
}

// ---- vmemd for the "d" runs: always --debug, so the log holds the broker's counts after every message ----

// socket paths hold at most 107 characters, so test files live in /tmp
inline const char *SOCK = "/tmp/vmemd-test.sock", *LOG = "/tmp/vmemd-test.log", *CONF = "/tmp/vmemd-test.toml";

// starts bin/vmemd with this config text; returns its pid, or -1 if it did not come up within 1 s
inline pid_t start_daemon(const char *conf) {
    unlink(LOG);
    if (FILE *f = fopen(CONF, "w")) fputs(conf, f), fclose(f);
    pid_t pid = fork();
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(127);
        execl("bin/vmemd", "vmemd", "--socket", SOCK, "--log", LOG, "--config", CONF, "--debug", (char *)nullptr);
        _exit(127);
    }
    sockaddr_un a = {};
    a.sun_family = AF_UNIX;
    strcpy(a.sun_path, SOCK);
    for (int i = 0; i < 100; i++) {
        if (waitpid(pid, nullptr, WNOHANG) == pid) return -1;
        int s = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        bool up = connect(s, (sockaddr *)&a, sizeof a) == 0;
        close(s);
        if (up) return pid;
        usleep(10000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    return -1;
}

inline void stop_daemon(pid_t pid) {
    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
    unlink(SOCK);
}

// ---- running a test body ----

// backend letter: c cuda, u uvm, v vmem without a daemon, d vmem with a vmemd this function starts
// the backend and the daemon socket come from the letter, not from VMEM_BACKEND or VMEM_DAEMON
// the broker's host limit (512 MiB) is small, so test 2's out-of-memory step hits its refusal before the machine's limit
template <class F> int run(const char *name, const char *letter, F body) {
    std::string l = letter ? letter : "";
    if (l != "c" && l != "u" && l != "v" && l != "d") {
        fprintf(stderr, "usage: %s c|u|v|d   (cuda, uvm, vmem, vmem with vmemd)\n", name);
        return 2;
    }
    bool daemon = l == "d";
    vmem::Kind kind = l == "c" ? vmem::Kind::Cuda : l == "u" ? vmem::Kind::Uvm : vmem::Kind::Vmem;
    pid_t pid = daemon ? start_daemon("vram_limit_mib = 5000\ndram_limit_mib = 512\n") : 0;
    if (pid < 0) {
        printf("FAIL: vmemd did not start\n");
        return 2;
    }
    unsetenv("VMEM_BACKEND");
    unsetenv("VMEM_DAEMON");
    int rc = 0;
    try {
        vmem::init({.backend = kind, .daemon = daemon ? SOCK : nullptr, .name = (std::string(name) + l).c_str()});
        printf("backend: %s\n", vmem::get().info().backend);
        expect(vmem::get().info().link == (daemon ? vmem::Link::Connected : vmem::Link::Local),
               daemon ? "link to vmemd is connected" : "no vmemd: link is local");
        body();
        // a broker lost mid-test turns the link Broken and the backend goes on locally, so every check above could
        // pass without the daemon. this makes sure the whole body really ran against it
        if (daemon) expect(vmem::get().info().link == vmem::Link::Connected, "link to vmemd still connected at the end");
        vmem::shutdown();
    } catch (const vmem::Error &e) {
        printf("FAIL: uncaught %s (%s)\n", e.what(), vmem::status_name(e.status));
        rc = 2;
    }
    if (daemon) {
        stop_daemon(pid);
        printf("vmemd log: %s\n", LOG);
    }
    if (rc) return rc;
    printf(fails ? "FAIL: %d check(s)\n" : "PASS\n", fails);
    return fails != 0;
}

// ---- line protocol of the daemon test clients (state_client, pair_client) ----

constexpr size_t MiB = 1 << 20;

// own counts in MiB, plus the whole machine's when the client knows it. without a broker the line starts with
// "local", so the checker does not look for it in the log
inline void show(long all_d = -1, long all_h = -1) {
    bool on = vmem::get().info().link == vmem::Link::Connected;
    printf("%s %zu %zu", on ? "view" : "local", vmem::get().used(vmem::Loc::Device) >> 20, vmem::get().used(vmem::Loc::Host) >> 20);
    if (all_d >= 0) printf(" %ld %ld", all_d, all_h);
    printf("\n");
    fflush(stdout);
}

// check own counts, then show them
inline void step(size_t d, size_t h, const char *what, long all_d = -1, long all_h = -1) {
    expect(vmem::get().used(vmem::Loc::Device) >> 20 == d && vmem::get().used(vmem::Loc::Host) >> 20 == h, what);
    show(all_d, all_h);
}

// "WAIT what", then block until the checker sends a line
inline void wait_for(const char *what) {
    printf("WAIT %s\n", what);
    fflush(stdout);
    char line[64];
    if (!fgets(line, sizeof line, stdin)) exit(2);
}

inline void *chunk() { return vmem::alloc(4 * MiB); }   // one 4 MiB chunk

}  // namespace t
