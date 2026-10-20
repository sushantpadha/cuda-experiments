// vmem backend: it uses CUDA VMM chunks, place on device first, then on pinned host (HOST_NUMA) when the device is full
// - each chunk = one address reservation + one cuMemCreate handle mapped over all of it
// - move swaps the handle under the same address; on failure the chunk keeps its old handle
// - local mode: device until cuMemCreate fails, then host
// - daemon mode: ask vmemd for the location on create and for permission on move (protocol.h, daemon/README.md)
// - no broker at open: local mode. broker lost later: link Broken, local mode until reconnect() resyncs
// TODO: MPS support

// mostly adapted from primitives/src/vmem.cuh
// differences: we are not tracking reservations seperately -- that level of detail is not provided to the consumer of the library anyway

// TODO: Note to self: This code would look so much better with monadic types :/

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

/*
This backend implements the core functionality of libvmem!
- That is, VMM chunks and the VMM remap protocol.
- Ideally, we would allow reserve_va/unreserve_va, but that is not yet implemented.
- It can operate in Local mode --- assign chunks independently of the broker.
- It can operate in Daemon mode --- ask the broker for 2 things: the location on create and for permission on move.
- Broker can allow reusing VRAM+DRAM more efficiently by tracking all allocation sizes
- On any such "decision" point, check if daemon mode; if broker connected, good; else, attempt reconnect; else (if local mode) proceed.
*/
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

        // ! enter daemon mode if a broker answers; none running means local mode
        if (!c.daemon) return Status::Ok;
        path_ = c.daemon;
        name_ = c.name ? c.name : program_invocation_short_name;
        int e = connect_broker();  // this call sets link_ if successful
        if (e == 0 || e == ENOENT || e == ECONNREFUSED) return Status::Ok;
        err_ = "vmemd " + path_ + ": " + strerror(e);
        return Status::Disconnected;
    }
    void close() noexcept override { disconnect_broker(Link::Local); } // explicit disconnect -> back to local mode
    const char *name() const override { return "vmem"; }
    size_t granularity() const override { return gran_; }
    bool can_move() const override { return true; }

    // ---- memory ----

    // consumer API: first generate address, then send ALLOC packet to broker for permission
    Status create(size_t size, Chunk &out) override {
        // reserve va
        // TODO: expose reserve_va/unreserve_va for more flexibility when using vmem backend; v2
        CUdeviceptr va = 0;
        if (Status s = cu(cuMemAddressReserve(&va, size, gran_, 0, 0), "cuMemAddressReserve"); s != Status::Ok)
            return s;
        Loc loc = Loc::Device;
        bool counted = false;   // the broker has this chunk in its counts
        if (Status s = decide_place(va, size, loc, counted); s != Status::Ok) {
            cuMemAddressFree(va, size);
            return s;
        }
        CUmemGenericAllocationHandle h;
        Status s = _create(size, loc, h);
        // ! broker's advice failed! go ahead anyway and inform them via ALLOCFAIL
        if (s == Status::NoMemory && loc == Loc::Device) {
            s = _create(size, loc = Loc::Host, h);
            if (s == Status::Ok && counted)
                send_msg({proto::Type::AllocFail, loc, va});
        }
        // good path: map
        if (s == Status::Ok) {
            s = map(va, size, h);
            if (s != Status::Ok) cuMemRelease(h);
        }
        // bad path: need to free va and update broker state
        if (s != Status::Ok) {
            if (counted) send_msg({proto::Type::Free, loc, va}); // the broker counted it already, tell it that we failed
            cuMemAddressFree(va, size);
            return s;
        }
        // good path final: update state
        chunks_[va] = {h, size, loc};
        used_[(int)loc] += size;
        out = {(void *)va, size, loc};
        return Status::Ok;
    }

    // consumer API: unmap, release, free va
    Status destroy(Chunk &c) override {
        CUdeviceptr va = (CUdeviceptr)c.ptr;
        Status s = cu(cuMemUnmap(va, c.size), "cuMemUnmap");
        if (s == Status::Ok) s = cu(cuMemRelease(chunks_.at(va).h), "cuMemRelease");
        if (s == Status::Ok) s = cu(cuMemAddressFree(va, c.size), "cuMemAddressFree");
        if (s != Status::Ok) return Status::Fatal;
        chunks_.erase(va);
        used_[(int)c.loc] -= c.size;
        send_msg({proto::Type::Free, c.loc, va}); // inform broker
        c = {};
        return Status::Ok;
    }

    // consumer API: in daemon mode: ask broker to grant the move first; then report with MoveDone
    // ! blocking
    Status move(Chunk &c, Loc to) override {
        if (c.loc == to) return Status::Ok;
        CUdeviceptr va = (CUdeviceptr)c.ptr;
        bool asked = false;
        if (link_ == Link::Connected) {
            proto::Reply rep;
            if (send_and_recv({proto::Type::Move, to, va}, rep)) {
                if (rep.status != Status::Ok) {
                    err_ = std::string("vmemd refused the move to ") + loc_name(to);
                    return Status::NoMemory;
                }
                asked = true;
            }
        }
        Status s = swap(c, to);
        if (asked) send_msg({proto::Type::MoveDone, c.loc, va});   // the target on success, the old loc on failure
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
        disconnect_broker(Link::Broken);
        if (int e = connect_broker(); e != 0) {
            err_ = "vmemd " + path_ + ": " + strerror(e);
            return Status::Disconnected;
        }
        for (auto &[va, c] : chunks_) send_msg({proto::Type::Resync, c.loc, va, c.size});
        return link_ == Link::Connected ? Status::Ok : Status::Disconnected;
    }
    size_t used() const override { return used_[0] + used_[1]; }
    size_t used(Loc l) const override { return used_[(int)l]; }
    const std::string &last_error() const override { return err_; }

private:
    // helper: ask the broker where a new chunk goes; without a link the answer is the device (create falls back to host)
    // NoMemory when the broker refuses. counted: the broker now has the chunk in its counts
    Status decide_place(CUdeviceptr va, size_t size, Loc &loc, bool &counted) {
        if (link_ != Link::Connected) return Status::Ok;
        proto::Reply rep;
        if (!send_and_recv({proto::Type::Alloc, Loc::Device, va, size}, rep)) return Status::Ok;   // link broke: local mode
        if (rep.status != Status::Ok) {
            err_ = "vmemd refused the allocation";
            return Status::NoMemory;
        }
        loc = rep.loc;
        counted = true;
        return Status::Ok;
    }

    // copy into a new handle through a scratch range, then swap it in under the same address
    // ! blocking; strong guarantee: on failure the chunk keeps its old handle, does not leak, return non-trivial status
    Status swap(Chunk &c, Loc to) {
        CUdeviceptr va = (CUdeviceptr)c.ptr;
        CUmemGenericAllocationHandle old = chunks_.at(va).h, nh;
        if (Status s = _create(c.size, to, nh); s != Status::Ok) return s;

        Status s = copy_into(nh, va, c.size);
        if (s == Status::Ok) s = replace_backing(va, c.size, old, nh); // good path
        if (s != Status::Ok) { // bad path
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

    // copy the chunk at va into nh; nh is mapped at a scratch range just for the copy
    // ! blocking; strong guarantee: always unmaps and releases scratch VA so caller doesnt have to clean up for it
    Status copy_into(CUmemGenericAllocationHandle nh, CUdeviceptr va, size_t size) {
        CUdeviceptr tmp = 0;
        Status s = cu(cuMemAddressReserve(&tmp, size, gran_, 0, 0), "cuMemAddressReserve");
        if (s == Status::Ok) s = map(tmp, size, nh);
        if (s == Status::Ok) {
            s = cu(cuMemcpyDtoD(tmp, va, size), "cuMemcpyDtoD");
            // ! FENCE
            if (s == Status::Ok) s = cu(cuCtxSynchronize(), "copy sync");
            cuMemUnmap(tmp, size);
        }
        if (tmp) cuMemAddressFree(tmp, size);
        return s;
    }

    // swap nh in under va; if it will not map, put old back. Fatal if even that fails (chunk lost its backing)
    // ! non-atomic swap! a crash here WILL be fatal
    Status replace_backing(CUdeviceptr va, size_t size, CUmemGenericAllocationHandle old, CUmemGenericAllocationHandle nh) {
        Status s = cu(cuMemUnmap(va, size), "cuMemUnmap");
        if (s != Status::Ok) return s; // cant unmap >_<
        s = map(va, size, nh);
        if (s == Status::Ok) return s;
        // fallback: put old back; failure means fatal
        return map(va, size, old) == Status::Ok ? s : Status::Fatal;
    }

    // ---- vmemd link ----

    // connect and say Hello; return 0 or the errno of the failed step
    int connect_broker() {
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
        // we have a connection; set link=Link::Connected and send hello
        fd_ = fd;
        link_ = Link::Connected;
        proto::Msg m = {proto::Type::Hello};
        strncpy(m.name, name_.c_str(), proto::NAME_LEN - 1);
        proto::Reply rep;
        return send_and_recv(m, rep) ? 0 : ETIMEDOUT;
    }

    // a failed send or recv breaks the link; the backend goes on in local mode; noexcept and multiple calls dont crash
    void disconnect_broker(Link next) noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        link_ = next;
    }

    // send a message; return false if failed
    bool send_msg(const proto::Msg &m) {
        if (link_ != Link::Connected) return false;
        // MSG_NOSIGNAL avoids SIGPIPE fatal crash
        if (::send(fd_, &m, sizeof m, MSG_NOSIGNAL) == (ssize_t)sizeof m) return true;
        err_ = std::string("vmemd send: ") + strerror(errno);
        disconnect_broker(Link::Broken); // ! set to broken right now
        return false;
    }

    // post message and listen for reply (matches on sizeof rep); return false if failed
    bool send_and_recv(const proto::Msg &m, proto::Reply &rep) {
        if (!send_msg(m)) return false;
        ssize_t n;
        do
            n = ::recv(fd_, &rep, sizeof rep, 0);
        while (n < 0 && errno == EINTR);
        if (n == (ssize_t)sizeof rep) return true;

        err_ = n < 0 ? std::string("vmemd recv: ") + strerror(errno) : "vmemd closed the connection";
        disconnect_broker(Link::Broken); // ! set to broken right now
        return false;
    }

    CUmemAllocationProp prop(Loc l) const {
        CUmemAllocationProp p = {};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type = l == Loc::Device ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA;
        p.location.id = l == Loc::Device ? dev_ : numa_;
        return p;
    }

    // helper: create handle for size at l
    Status _create(size_t size, Loc l, CUmemGenericAllocationHandle &h) {
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
    // ! changed in vmemd mode because we need to track handle, size, loc for Resync
    // rule of thumb: backend tracks detailed allocation information and relays to daemon; daemon tracks counters
    struct Rec {
        CUmemGenericAllocationHandle h;
        size_t size;
        Loc loc;
    };
    std::map<CUdeviceptr, Rec> chunks_;
    static constexpr int RECV_TIMEOUT_S = 3;
    std::string name_;   // sent in Hello
    std::string path_;   // vmemd socket; empty without a daemon
    int fd_ = -1;
    Link link_ = Link::Local;  // Local (local mode) or Connected (daemon mode, connected) or Broken (daemon mode, disconnected)
    std::string err_;
};

}  // namespace vmem
