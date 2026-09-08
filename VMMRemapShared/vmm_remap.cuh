#pragma once

// VMMVector + remap(): the cudaremap primitive dropped into the growable-vector
// from ../VMMVector/vmm_vector.cuh. remap(chunk_idx, to_host) moves one chunk's
// physical backing between device and host while d_ptr stays fixed.

#include "common.cuh"
#include "remap.cuh"
#include <cassert>

template<typename T>
class VMMRemapVector {
public:
    static const int device = 0;

    CUdeviceptr d_ptr;
    size_t size_;
    size_t capacity_;
    size_t max_capacity_;
    size_t chunk_size_;
    size_t chunk_fits_;
    CUmemAllocationProp props_;
    CUmemAccessDesc accessDesc_;
    std::vector<CUmemGenericAllocationHandle> handles_;
    std::vector<char> chunk_on_host_;   // per chunk: 1 == backed by HOST_NUMA

    CUmemGenericAllocationHandle create_chunk() {
        CUmemGenericAllocationHandle h;
        CU_CHECK( cuMemCreate(&h, chunk_size_, &props_, 0) );   // new chunks: device
        return h;
    }

    VMMRemapVector(size_t max_capacity, size_t chunk_size) {
        d_ptr = 0; size_ = 0; capacity_ = 0;
        max_capacity_ = max_capacity;
        accessDesc_ = {}; props_ = {};

        props_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        props_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props_.location.id = device;
        props_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        size_t gran = remap_granularity();
        chunk_size_ = ROUND_UP(chunk_size, gran);
        if (chunk_size_ < sizeof(T) || chunk_size_ % sizeof(T) != 0)
            throw std::runtime_error("bad chunk_size");
        chunk_fits_ = chunk_size_ / sizeof(T);
        max_capacity_ = ROUND_UP(max_capacity, chunk_fits_);

        accessDesc_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc_.location.id = device;
        accessDesc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CU_CHECK( cuMemAddressReserve(&d_ptr, max_capacity_ * sizeof(T), 0, 0, 0) );
        DPRINT("VMMRemapVector d_ptr=%p chunk_size=%zu chunk_fits=%zu",
               (void*)d_ptr, chunk_size_, chunk_fits_);
    }

    ~VMMRemapVector() {
        if (d_ptr && capacity_)
            CU_CHECK( cuMemUnmap(d_ptr, capacity_ * sizeof(T)) );
        for (auto h : handles_) CU_CHECK( cuMemRelease(h) );
        if (d_ptr && max_capacity_)
            CU_CHECK( cuMemAddressFree(d_ptr, max_capacity_ * sizeof(T)) );
    }

    void reserve(size_t new_capacity) {
        if (new_capacity <= capacity_) return;
        size_t nc = ROUND_UP(new_capacity, chunk_fits_);
        if (nc > max_capacity_) throw std::runtime_error("max_capacity exceeded");
        size_t extra = nc / chunk_fits_ - capacity_ / chunk_fits_;
        CUdeviceptr ptr = d_ptr + capacity_ * sizeof(T);
        for (size_t i = 0; i < extra; ++i) {
            auto h = create_chunk();
            CU_CHECK( cuMemMap(ptr, chunk_size_, 0, h, 0) );
            CU_CHECK( cuMemSetAccess(ptr, chunk_size_, &accessDesc_, 1) );
            handles_.push_back(h);
            chunk_on_host_.push_back(0);
            capacity_ += chunk_fits_;
            ptr += chunk_size_;
        }
    }

    void push_back(T value) {
        if (size_ == capacity_) reserve(size_ > 1 ? size_ * 2 : 2);
        CU_CHECK( cuMemcpyHtoD(d_ptr + size_ * sizeof(T), &value, sizeof(T)) );
        ++size_;
    }

    // move chunk `chunk_idx` to host (to_host=true) or back to device.
    // d_ptr + chunk_idx*chunk_size_ is unchanged; contents preserved.
    void remap(size_t chunk_idx, bool to_host) {
        assert(chunk_idx < handles_.size());
        if ((bool)chunk_on_host_[chunk_idx] == to_host) return;
        remap_backing(d_ptr + chunk_idx * chunk_size_, chunk_size_,
                      handles_[chunk_idx], to_host, accessDesc_,
                      props_.requestedHandleTypes);
        chunk_on_host_[chunk_idx] = to_host ? 1 : 0;
    }

    bool chunk_on_host(size_t i) const { return chunk_on_host_[i]; }
    size_t n_chunks() const { return handles_.size(); }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    size_t chunk_size() const { return chunk_size_; }
    T* data() { return (T*)d_ptr; }

    size_t copy_to_host(T* host_ptr) const {
        size_t sz = size_ * sizeof(T);
        CU_CHECK( cuMemcpyDtoH(host_ptr, d_ptr, sz) );
        return sz;
    }
};
