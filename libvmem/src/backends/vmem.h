// vmem backend: it uses CUDA VMM chunks, place on device first, then on pinned host (HOST_NUMA) when the device is full
// - each chunk = one address reservation + one cuMemCreate handle mapped over all of it
// - move swaps the handle under the same address; on failure the chunk keeps its old handle
// - local mode only: device until cuMemCreate fails, then host
// - daemon mode:
// TODO: vmemd client ask the broker for the location on create and for permission on move
// TODO: MPS support

// mostly adapted from primitives/src/vmem.cuh which was in turn adapted from experiments/VMMVector
// differences: we are not tracking reservations seperately -- that level of detail is not provided to the consumer of the library anyway

#pragma once

#include "../backend.h"

#include <cuda.h>
#include <cuda_runtime.h>

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
        return Status::Ok;
    }
    void close() noexcept override {}
    const char *name() const override { return "vmem"; }
    size_t granularity() const override { return gran_; }
    bool can_move() const override { return true; }

    // ---- memory ----

    Status create(size_t size, Chunk &out) override {
        CUdeviceptr va = 0;
        if (Status s = cu(cuMemAddressReserve(&va, size, gran_, 0, 0), "cuMemAddressReserve"); s != Status::Ok)
            return s;
        Loc loc = Loc::Device;
        CUmemGenericAllocationHandle h;
        Status s = make(size, loc, h);
        if (s == Status::NoMemory) s = make(size, loc = Loc::Host, h);
        if (s == Status::Ok && (s = map(va, size, h)) != Status::Ok) cuMemRelease(h);
        if (s != Status::Ok) {
            cuMemAddressFree(va, size);
            return s;
        }
        handles_[va] = h;
        used_[(int)loc] += size;
        out = {(void *)va, size, loc};
        return Status::Ok;
    }

    Status destroy(Chunk &c) override {
        CUdeviceptr va = (CUdeviceptr)c.ptr;
        Status s = cu(cuMemUnmap(va, c.size), "cuMemUnmap");
        if (s == Status::Ok) s = cu(cuMemRelease(handles_.at(va)), "cuMemRelease");
        if (s == Status::Ok) s = cu(cuMemAddressFree(va, c.size), "cuMemAddressFree");
        if (s != Status::Ok) return Status::Fatal;
        handles_.erase(va);
        used_[(int)c.loc] -= c.size;
        c = {};
        return Status::Ok;
    }

    // copy into a new handle through a scratch range, then swap it in under the same address
    Status move(Chunk &c, Loc to) override {
        if (c.loc == to) return Status::Ok;
        CUdeviceptr va = (CUdeviceptr)c.ptr, tmp = 0;
        CUmemGenericAllocationHandle old = handles_.at(va), nh;
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
        handles_[va] = nh;
        used_[(int)c.loc] -= c.size;
        used_[(int)to] += c.size;
        c.loc = to;
        return Status::Ok;
    }

    // ---- admin ----

    Link link() const override { return Link::Local; }
    Status reconnect() override { return Status::Unsupported; }
    size_t used() const override { return used_[0] + used_[1]; }
    size_t used(Loc l) const override { return used_[(int)l]; }
    const std::string &last_error() const override { return err_; }

private:
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
    std::map<CUdeviceptr, CUmemGenericAllocationHandle> handles_;   // chunk address -> its handle
    std::string err_;
};

}  // namespace vmem
