// cuda backend: plain cudaMalloc baseline
// device only, no host fallback, chunks never move
#pragma once

#include "../backend.h"

#include <cuda_runtime.h>

namespace vmem {

class CudaBackend : public Backend {
public:
    Status open(const BackendConfig &c) override {
        if (!ok(cudaSetDevice(c.device), "cudaSetDevice")) return Status::Fatal;
        return ok(cudaFree(nullptr), "context init") ? Status::Ok : Status::Fatal;
    }
    void close() noexcept override {}
    const char *name() const override { return "cuda"; }
    size_t granularity() const override { return 2 << 20; }   // same as VMM, so backends get equal sizes
    bool can_move() const override { return false; }

    Status create(size_t size, Chunk &out) override {
        void *p = nullptr;
        cudaError_t e = cudaMalloc(&p, size);
        if (e == cudaErrorMemoryAllocation) {
            cudaGetLastError();   // not sticky; clear it
            err_ = "cudaMalloc: out of memory";
            return Status::NoMemory;
        }
        if (!ok(e, "cudaMalloc")) return Status::Fatal;
        out = {p, size, Loc::Device};
        used_ += size;
        return Status::Ok;
    }

    Status destroy(Chunk &c) override {
        if (!ok(cudaFree(c.ptr), "cudaFree")) return Status::Fatal;
        used_ -= c.size;
        c = {};
        return Status::Ok;
    }

    Status move(Chunk &c, Loc to) override { return c.loc == to ? Status::Ok : Status::Unsupported; }

    Link link() const override { return Link::Local; }
    Status reconnect() override { return Status::Unsupported; }
    size_t used() const override { return used_; }
    size_t used(Loc l) const override { return l == Loc::Device ? used_ : 0; }
    const std::string &last_error() const override { return err_; }

private:
    bool ok(cudaError_t e, const char *what) {
        if (e == cudaSuccess) return true;
        err_ = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }

    size_t used_ = 0;
    std::string err_;
};

}  // namespace vmem
