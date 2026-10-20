// vmem backend: it uses CUDA VMM chunks, place on device first, then on pinned host (HOST_NUMA) when the device is full
// - each chunk = one address reservation + one cuMemCreate handle mapped over all of it
// - move swaps the handle under the same address; on failure the chunk keeps its old handle
// - local mode: device until cuMemCreate fails, then host
// - daemon mode: ask vmemd for the location on create and for permission on move (protocol.h, daemon/README.md)
// - no broker at open: local mode. broker lost later: link Broken, local mode until reconnect() resyncs
// TODO: MPS support

// mostly adapted from primitives/src/vmem.cuh which was in turn adapted from experiments/VMMVector
// differences: we are not tracking reservations seperately -- that level of detail is not provided to the consumer of the library anyway

#pragma once

#include "../backend.h"
#include "../protocol.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstring>
#include <map>

namespace vmem {

class VmemBackend : public Backend {
public:
    Status open(const BackendConfig &c) override {
        dev_ = c.device;
        // runtime calls create the primary context and make it current, same as the other backends
        if (cudaSetDevice(dev_) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
            err_ = "context init failed";
            return Status::Fatal;
        }
        int vmm = 0, host_vmm = 0, numa = -1;
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev_);
        cuDeviceGetAttribute(&host_vmm, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev_);
        cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev_);
        if (!vmm || !host_vmm) {
            err_ = "device does not support VMM on device and host";
            return Status::Unsupported;
        }
        numa_ = numa < 0 ? 0 : numa;

        // same granularity on both sides, so a move keeps the size
        size_t gd = 0, gh = 0;
        CUmemAllocationProp pd = prop(Loc::Device), ph = prop(Loc::Host);
        if (Status s = cu(cuMemGetAllocationGranularity(&gd, &pd, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
            s != Status::Ok)
            return s;
        if (Status s = cu(cuMemGetAllocationGranularity(&gh, &ph, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
            s != Status::Ok)
            return s;
        gran_ = gd > gh ? gd : gh;

        // daemon mode if a broker answers; none running means local mode
        if (!c.daemon) return Status::Ok;
        path_ = c.daemon;
        int e = dial();
        if (e == 0 || e == ENOENT || e == ECONNREFUSED) return Status::Ok;
        err_ = "vmemd " + path_ + ": " + strerror(e);
        return Status::Disconnected;
    }
    void close() noexcept override { hang_up(Link::Local); }
    const char *name() const override { return "vmem"; }
    size_t granularity() const override { return gran_; }
    bool can_move() const override { return true; }

    // ---- memory ----

    // the address comes first: it names the chunk in Alloc
    Status create(size_t size, Chunk &out) override {
        CUdeviceptr va = 0;
        if (Status s = cu(cuMemAddressReserve(&va, size, gran_, 0, 0), "cuMemAddressReserve"); s != Status::Ok)
            return s;
        Loc loc = Loc::Device;
        bool asked = false;
        if (link_ == Link::Connected) {
            proto::Reply rep;
            if (ask({proto::Type::Alloc, Loc::Device, va, size}, rep)) {
                if (rep.status != Status::Ok) {
                    cuMemAddressFree(va, size);
                    err_ = "vmemd refused the allocation";
                    return Status::NoMemory;
                }
                loc = rep.loc;
                asked = true;
            }
        }
        CUmemGenericAllocationHandle h;
        Status s = make(size, loc, h);
        if (s == Status::NoMemory && loc == Loc::Device) {
            s = make(size, loc = Loc::Host, h);
            if (s == Status::Ok && asked) post({proto::Type::AllocFail, loc, va});
        }
        if (s == Status::Ok && (s = map(va, size, h)) != Status::Ok) cuMemRelease(h);
        if (s != Status::Ok) {
            if (asked) post({proto::Type::Free, loc, va});   // the broker counted it already
            cuMemAddressFree(va, size);
            return s;
        }
        chunks_[va] = {h, size, loc};
        used_[(int)loc] += size;
        out = {(void *)va, size, loc};
        return Status::Ok;
    }

    Status destroy(Chunk &c) override {
        CUdeviceptr va = (CUdeviceptr)c.ptr;
        Status s = cu(cuMemUnmap(va, c.size), "cuMemUnmap");
        if (s == Status::Ok) s = cu(cuMemRelease(chunks_.at(va).h), "cuMemRelease");
        if (s == Status::Ok) s = cu(cuMemAddressFree(va, c.size), "cuMemAddressFree");
        if (s != Status::Ok) return Status::Fatal;
        chunks_.erase(va);
        used_[(int)c.loc] -= c.size;
        post({proto::Type::Free, c.loc, va});
        c = {};
        return Status::Ok;
    }

    // vmemd grants the move first; MoveDone then reports where the chunk ended up
    Status move(Chunk &c, Loc to) override {
        if (c.loc == to) return Status::Ok;
        CUdeviceptr va = (CUdeviceptr)c.ptr;
        bool asked = false;
        if (link_ == Link::Connected) {
            proto::Reply rep;
            if (ask({proto::Type::Move, to, va}, rep)) {
                if (rep.status != Status::Ok) {
                    err_ = std::string("vmemd refused the move to ") + loc_name(to);
                    return Status::NoMemory;
                }
                asked = true;
            }
        }
        Status s = swap(c, to);
        if (asked) post({proto::Type::MoveDone, c.loc, va});   // the target on success, the old loc on failure
        return s;
    }

    // ---- admin ----

    Link link() const override { return link_; }

    // after the broker restarts: Hello, then one Resync per live chunk
    Status reconnect() override {
        if (path_.empty()) {
            err_ = "no vmemd socket configured";
            return Status::Unsupported;
        }
        hang_up(Link::Broken);
        if (int e = dial(); e != 0) {
            err_ = "vmemd " + path_ + ": " + strerror(e);
            return Status::Disconnected;
        }
        for (auto &[va, c] : chunks_) post({proto::Type::Resync, c.loc, va, c.size});
        return link_ == Link::Connected ? Status::Ok : Status::Disconnected;
    }
    size_t used() const override { return used_[0] + used_[1]; }
    size_t used(Loc l) const override { return used_[(int)l]; }
    const std::string &last_error() const override { return err_; }

private:
    // copy into a new handle through a scratch range, then swap it in under the same address
    Status swap(Chunk &c, Loc to) {
        CUdeviceptr va = (CUdeviceptr)c.ptr, tmp = 0;
        CUmemGenericAllocationHandle old = chunks_.at(va).h, nh;
        if (Status s = make(c.size, to, nh); s != Status::Ok) return s;

        // 1. copy; the allocator has synchronized, so nothing else runs
        Status s = cu(cuMemAddressReserve(&tmp, c.size, gran_, 0, 0), "cuMemAddressReserve");
        if (s == Status::Ok && (s = map(tmp, c.size, nh)) == Status::Ok) {
            s = cu(cuMemcpyDtoD(tmp, va, c.size), "cuMemcpyDtoD");
            if (s == Status::Ok) s = cu(cuCtxSynchronize(), "copy sync");
            cuMemUnmap(tmp, c.size);
        }
        if (tmp) cuMemAddressFree(tmp, c.size);

        // 2. swap; if the new handle will not map, put the old one back
        if (s == Status::Ok && (s = cu(cuMemUnmap(va, c.size), "cuMemUnmap")) == Status::Ok &&
            (s = map(va, c.size, nh)) != Status::Ok && map(va, c.size, old) != Status::Ok)
            s = Status::Fatal;   // chunk lost its backing
        if (s != Status::Ok) {
            cuMemRelease(nh);
            return s;
        }
        cuMemRelease(old);
        chunks_[va] = {nh, c.size, to};
        used_[(int)c.loc] -= c.size;
        used_[(int)to] += c.size;
        c.loc = to;
        return Status::Ok;
    }

    // ---- vmemd link ----

    // connect and say Hello; 0 or the errno of the failed step
    int dial() {
        sockaddr_un a = {};
        a.sun_family = AF_UNIX;
        if (path_.size() >= sizeof a.sun_path) return ENAMETOOLONG;
        strcpy(a.sun_path, path_.c_str());
        timeval tv = {RECV_TIMEOUT_S, 0};
        int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        if (fd < 0) return errno;
        if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) || connect(fd, (sockaddr *)&a, sizeof a)) {
            int e = errno;
            ::close(fd);
            return e;
        }
        fd_ = fd;
        link_ = Link::Connected;
        proto::Msg m = {proto::Type::Hello};
        strncpy(m.name, program_invocation_short_name, proto::NAME_LEN - 1);
        proto::Reply rep;
        return ask(m, rep) ? 0 : ETIMEDOUT;
    }

    // a failed send or recv breaks the link; the backend goes on in local mode
    void hang_up(Link next) noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        link_ = next;
    }

    bool post(const proto::Msg &m) {
        if (link_ != Link::Connected) return false;
        if (send(fd_, &m, sizeof m, MSG_NOSIGNAL) == (ssize_t)sizeof m) return true;
        err_ = std::string("vmemd send: ") + strerror(errno);
        hang_up(Link::Broken);
        return false;
    }

    bool ask(const proto::Msg &m, proto::Reply &rep) {
        if (!post(m)) return false;
        ssize_t n;
        do n = recv(fd_, &rep, sizeof rep, 0); while (n < 0 && errno == EINTR);
        if (n == (ssize_t)sizeof rep) return true;
        err_ = n < 0 ? std::string("vmemd recv: ") + strerror(errno) : "vmemd closed the connection";
        hang_up(Link::Broken);
        return false;
    }

    CUmemAllocationProp prop(Loc l) const {
        CUmemAllocationProp p = {};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type = l == Loc::Device ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA;
        p.location.id = l == Loc::Device ? dev_ : numa_;
        return p;
    }

    Status make(size_t size, Loc l, CUmemGenericAllocationHandle &h) {
        CUmemAllocationProp p = prop(l);
        return cu(cuMemCreate(&h, size, &p, 0), l == Loc::Device ? "cuMemCreate device" : "cuMemCreate host");
    }

    // map and give the device read/write (host chunks are read over PCIe); unmaps again on failure
    Status map(CUdeviceptr va, size_t size, CUmemGenericAllocationHandle h) {
        if (Status s = cu(cuMemMap(va, size, 0, h, 0), "cuMemMap"); s != Status::Ok) return s;
        CUmemAccessDesc a = {};
        a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        a.location.id = dev_;
        a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        Status s = cu(cuMemSetAccess(va, size, &a, 1), "cuMemSetAccess");
        if (s != Status::Ok) cuMemUnmap(va, size);
        return s;
    }

    // out of memory is not sticky; anything else is treated as fatal
    Status cu(CUresult r, const char *what) {
        if (r == CUDA_SUCCESS) return Status::Ok;
        const char *n = "?";
        cuGetErrorName(r, &n);
        err_ = std::string(what) + ": " + n;
        return r == CUDA_ERROR_OUT_OF_MEMORY ? Status::NoMemory : Status::Fatal;
    }

    int dev_ = 0, numa_ = 0;
    size_t gran_ = 0;
    size_t used_[2] = {};   // indexed by Loc
    struct Rec {
        CUmemGenericAllocationHandle h;
        size_t size;
        Loc loc;
    };
    std::map<CUdeviceptr, Rec> chunks_;   // chunk address -> its handle, size and location (for Resync)
    static constexpr int RECV_TIMEOUT_S = 2;   // a hung broker cannot hang the app
    std::string path_;   // vmemd socket; empty without a daemon
    int fd_ = -1;
    Link link_ = Link::Local;
    std::string err_;
};

}  // namespace vmem
