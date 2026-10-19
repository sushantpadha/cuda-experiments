// uvm backend: cudaMallocManaged, the driver migrates pages on faults
// ! placement is handled by the driver, so used(Loc) is unsupported
#pragma once

#include "../backend.h"

#include <cuda_runtime.h>

namespace vmem {

class UvmBackend : public Backend {
public:
    Status open(const BackendConfig &c) override {
        dev_ = c.device;
        int managed = 0;
        if (!ok(cudaSetDevice(dev_), "cudaSetDevice") ||
            !ok(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, dev_), "managed memory attribute"))
            return Status::Fatal;
        if (!managed) {
            err_ = "device does not support managed memory";
            return Status::Unsupported;
        }
        return ok(cudaFree(nullptr), "context init") ? Status::Ok : Status::Fatal;
    }
    void close() noexcept override {}
    const char *name() const override { return "uvm"; }
    size_t granularity() const override { return 2 << 20; }   // same as VMM, so backends get equal sizes
    bool can_move() const override { return true; }

    Status create(size_t size, Chunk &out) override {
        void *p = nullptr;
        if (Status s = fail(cudaMallocManaged(&p, size), "cudaMallocManaged"); s != Status::Ok) return s;
        out = {p, size, Loc::Device};   // nominal: pages appear where first touched
        used_ += size;
        return Status::Ok;
    }

    Status destroy(Chunk &c) override {
        if (!ok(cudaFree(c.ptr), "cudaFree")) return Status::Fatal;
        used_ -= c.size;
        c = {};
        return Status::Ok;
    }

    // ! blocking prefetch; Ok means "asked and done", not "every page is there"
    // TODO: add cudaMemAdvise preferred location if pages drift back and the hint should stick
    Status move(Chunk &c, Loc to) override {
        if (c.loc == to) return Status::Ok;
        cudaMemLocation l = {};
        l.type = to == Loc::Device ? cudaMemLocationTypeDevice : cudaMemLocationTypeHost;
        l.id = to == Loc::Device ? dev_ : 0;
        Status s = fail(cudaMemPrefetchAsync(c.ptr, c.size, l, 0, 0), "cudaMemPrefetchAsync");
        if (s == Status::Ok) s = fail(cudaStreamSynchronize(0), "prefetch sync");
        if (s == Status::Ok) c.loc = to;
        return s;
    }

    Link link() const override { return Link::Local; }
    Status reconnect() override { return Status::Unsupported; }
    size_t used() const override { return used_; }
    size_t used(Loc) const override { throw Error(Status::Unsupported, "uvm: the driver decides where pages live"); }
    const std::string &last_error() const override { return err_; }

private:
    bool ok(cudaError_t e, const char *what) {
        if (e == cudaSuccess) return true;
        err_ = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }

    // out of memory is not sticky; anything else is treated as fatal
    Status fail(cudaError_t e, const char *what) {
        if (ok(e, what)) return Status::Ok;
        if (e != cudaErrorMemoryAllocation) return Status::Fatal;
        cudaGetLastError();
        return Status::NoMemory;
    }

    int dev_ = 0;
    size_t used_ = 0;
    std::string err_;
};

}  // namespace vmem
