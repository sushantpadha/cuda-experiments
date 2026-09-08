#pragma once

// VMMVector + remap(): move one chunk's physical backing between
// CU_MEM_LOCATION_TYPE_DEVICE and CU_MEM_LOCATION_TYPE_HOST_NUMA while keeping
// the same virtual address. Trimmed copy of ../VMMVector/vmm_vector.cuh; the
// growable-vector machinery is kept so remap() drops straight into the original.

#include "common.cuh"
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
    int retain_last_;
    CUmemAllocationProp props_;
    CUmemAccessDesc accessDesc_;
    std::vector<CUmemGenericAllocationHandle> handles_;
    std::vector<char> chunk_on_host_;   // per chunk: 1 == backed by HOST_NUMA

    // props for a chunk backed on host vs device; everything else copied from props_
    CUmemAllocationProp props_for(bool host) const {
        CUmemAllocationProp p = props_;
        p.location.type = host ? CU_MEM_LOCATION_TYPE_HOST_NUMA
                               : CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = 0;  // NUMA node 0 / device 0 — coincide on this box
        return p;
    }

    auto push_chunk() {
        CUmemGenericAllocationHandle handle;
        CUmemAllocationProp p = props_for(false);   // new chunks start on device
        CU_CHECK( cuMemCreate(&handle, chunk_size_, &p, 0) );
        handles_.push_back(handle);
        chunk_on_host_.push_back(0);
        capacity_ += chunk_fits_;
        return handle;
    }

    void pop_chunk() {
        CU_CHECK( cuMemRelease(handles_.back()) );
        handles_.pop_back();
        chunk_on_host_.pop_back();
        capacity_ -= chunk_fits_;
    }

    void map_chunk(CUdeviceptr ptr, CUmemGenericAllocationHandle handle) {
        CU_CHECK( cuMemMap(ptr, chunk_size_, 0, handle, 0) );
    }
    void unmap_chunk(CUdeviceptr ptr) {
        CU_CHECK( cuMemUnmap(ptr, chunk_size_) );
    }
    void set_access(CUdeviceptr ptr) {
        // device access descriptor works for both device- and host-backed
        // chunks: the copy engine / kernels reach HOST_NUMA memory over the bus.
        CU_CHECK( cuMemSetAccess(ptr, chunk_size_, &accessDesc_, 1) );
    }

    VMMRemapVector(size_t max_capacity, size_t chunk_size, int retain_last = 0) {
        d_ptr = 0; size_ = 0; capacity_ = 0;
        max_capacity_ = max_capacity; chunk_size_ = chunk_size;
        accessDesc_ = {}; props_ = {}; retain_last_ = retain_last;

        props_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        props_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props_.location.id = device;
        props_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        // granularity: take the max over device and host so one chunk_size fits both
        size_t g = 0, gh = 0;
        CUmemAllocationProp hp = props_for(true), dp = props_for(false);
        CU_CHECK( cuMemGetAllocationGranularity(&g,  &dp, CU_MEM_ALLOC_GRANULARITY_MINIMUM) );
        CU_CHECK( cuMemGetAllocationGranularity(&gh, &hp, CU_MEM_ALLOC_GRANULARITY_MINIMUM) );
        size_t gran = g > gh ? g : gh;
        DPRINT("granularity device=%zu host=%zu -> %zu", g, gh, gran);

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
            auto h = push_chunk();
            map_chunk(ptr, h);
            set_access(ptr);
            ptr += chunk_size_;
        }
    }

    void push_back(T value) {
        if (size_ == capacity_) reserve(size_ > 1 ? size_ * 2 : 2);
        CU_CHECK( cuMemcpyHtoD(d_ptr + size_ * sizeof(T), &value, sizeof(T)) );
        ++size_;
    }

    void resize(size_t n) {
        if (n <= size_) { size_ = n; return; }
        if (n > capacity_) reserve(n);
        for (size_t i = size_; i < n; ++i) push_back(T{});
    }

    // --- the new bit -------------------------------------------------------
    // Move chunk `chunk_idx` to host (to_host=true) or back to device.
    // Virtual address d_ptr + chunk_idx*chunk_size_ is unchanged; contents kept.
    void remap(size_t chunk_idx, bool to_host) {
        assert(chunk_idx < handles_.size());
        if ((bool)chunk_on_host_[chunk_idx] == to_host) return;

        CUdeviceptr base = d_ptr + chunk_idx * chunk_size_;

        // 1. new physical backing at the target location
        CUmemAllocationProp p = props_for(to_host);
        CUmemGenericAllocationHandle nh;
        CU_CHECK( cuMemCreate(&nh, chunk_size_, &p, 0) );

        // 2. stage it on a scratch VA and copy the live contents into it
        CUdeviceptr scratch;
        CU_CHECK( cuMemAddressReserve(&scratch, chunk_size_, 0, 0, 0) );
        CU_CHECK( cuMemMap(scratch, chunk_size_, 0, nh, 0) );
        CU_CHECK( cuMemSetAccess(scratch, chunk_size_, &accessDesc_, 1) );
        CU_CHECK( cuMemcpyDtoD(scratch, base, chunk_size_) );  // bus copy if host-backed
        CU_CHECK( cuCtxSynchronize() );

        // 3. swap the backing under the original VA
        CU_CHECK( cuMemUnmap(base, chunk_size_) );
        CU_CHECK( cuMemRelease(handles_[chunk_idx]) );
        CU_CHECK( cuMemMap(base, chunk_size_, 0, nh, 0) );
        CU_CHECK( cuMemSetAccess(base, chunk_size_, &accessDesc_, 1) );

        // 4. drop the scratch VA (nh stays alive: still mapped at base)
        CU_CHECK( cuMemUnmap(scratch, chunk_size_) );
        CU_CHECK( cuMemAddressFree(scratch, chunk_size_) );

        handles_[chunk_idx] = nh;
        chunk_on_host_[chunk_idx] = to_host ? 1 : 0;
    }

    bool chunk_on_host(size_t i) const { return chunk_on_host_[i]; }
    size_t n_chunks() const { return handles_.size(); }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    size_t chunk_size() const { return chunk_size_; }
    size_t chunk_fits() const { return chunk_fits_; }
    T* data() { return (T*)d_ptr; }

    size_t copy_to_host(T* host_ptr) const {
        size_t sz = size_ * sizeof(T);
        CU_CHECK( cuMemcpyDtoH(host_ptr, d_ptr, sz) );
        return sz;
    }
};
