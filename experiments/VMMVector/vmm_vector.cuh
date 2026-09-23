#pragma once

#include "common.cuh"

template<typename T>
class VMMVector {
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

    auto push_chunk() {
        CUmemGenericAllocationHandle handle;
        CU_CHECK( cuMemCreate(&handle, chunk_size_, &props_, 0) );
        handles_.push_back(handle);
        capacity_ += chunk_fits_;
        return handle;
    }

    auto pop_chunk() {
        CUmemGenericAllocationHandle handle = handles_.back();
        CU_CHECK( cuMemRelease(handle) );
        handles_.pop_back();
        capacity_ -= chunk_fits_;
    }

    void map_chunk(CUdeviceptr ptr, CUmemGenericAllocationHandle handle) {
        CU_CHECK( cuMemMap(ptr, chunk_size_, 0, handle, 0) );
    }

    void unmap_chunk(CUdeviceptr ptr) {
        CU_CHECK( cuMemUnmap(ptr, chunk_size_) );
    }

    void set_access(CUdeviceptr ptr) {
        CU_CHECK( cuMemSetAccess(ptr, chunk_size_, &accessDesc_, 1) );
    }

    VMMVector(size_t max_capacity, size_t chunk_size, int retain_last = 0, bool use_device = true) {
        this->d_ptr = 0;
        this->size_ = 0;
        this->capacity_ = 0;
        this->max_capacity_ = max_capacity;
        this->chunk_size_ = chunk_size;
        this->accessDesc_ = {};
        this->props_ = {};
        this->handles_ = {};
        this->retain_last_ = retain_last;

        // pinned device allocation; fd-handled
        props_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        props_.location.type = use_device ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA;
        props_.location.id = device; // ! works because NUMA ID of GPU's associated CPU is also 0
        props_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        size_t min_granularity = 0;
        CU_CHECK( cuMemGetAllocationGranularity(&min_granularity, &props_, CU_MEM_ALLOC_GRANULARITY_MINIMUM) );
        DPRINT("min_granularity: %zu", min_granularity);
        chunk_size_ = ROUND_UP(chunk_size, min_granularity);

        if (chunk_size_ == 0) throw std::runtime_error("chunk_size must be > 0");
        if (chunk_size_ < sizeof(T)) throw std::runtime_error("chunk_size must be >= sizeof(T)");
        if (chunk_size_ % sizeof(T) != 0) throw std::runtime_error("chunk_size must be multiple of sizeof(T)");

        chunk_fits_ = chunk_size_ / sizeof(T);
        max_capacity_ = ROUND_UP(max_capacity, chunk_fits_);

        // access rights
        accessDesc_.location.type = use_device ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA;
        accessDesc_.location.id = device; // ! works because NUMA ID of GPU's associated CPU is also 0
        accessDesc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // reserve VA space
        CU_CHECK( cuMemAddressReserve(&d_ptr, max_capacity_ * sizeof(T), 0, 0, 0) );

        DPRINT("VMMVector: max_capacity=%zu chunk_size=%zu (in granularity) chunk_fits=%zu d_ptr=%p capacity=%zu size=%zu",
               max_capacity_, chunk_size_ / min_granularity, chunk_fits_, (void*)d_ptr, capacity_, size_);
    }

    ~VMMVector() {
        // unmap > free PAs > free VAs
        if (d_ptr && capacity_) {
            CU_CHECK( cuMemUnmap(d_ptr, capacity_ * sizeof(T)) );
            DPRINT("unmapped %zu chunks or %zu bytes", capacity_ / chunk_fits_, capacity_ * sizeof(T));
        }
        for (auto handle : handles_) {
            CU_CHECK( cuMemRelease(handle) );
            DPRINT("released PA %p", (void*)handle);
        }
        if (d_ptr && max_capacity_) {
            CU_CHECK( cuMemAddressFree(d_ptr, max_capacity_ * sizeof(T)) );
            DPRINT("freed VA %p", (void*)d_ptr);
        }
    }

    // inc size_, add new chunk if needed, copy value
    void push_back(T value) {
        // TODO: keep host-side cache and write whole chunks
        if (size_ == capacity_) reserve(size_ > 1 ? size_ * 2 : 2);
        // DPRINT("push_back: size=%zu capacity=%zu", size_, capacity_);
        CU_CHECK( cuMemcpyHtoD(d_ptr + size_ * sizeof(T), &value, sizeof(T)) );
        ++size_;
    }

    // dec size_, remove last chunk if possible
    void pop_back() {
        if (size_ == 0) return;

        --size_;

        // retain all chunks
        if (retain_last_ == -1) {
            return;
        }

        // retain atmost last `retain_last_` chunks
        if (capacity_ - size_ >= (retain_last_ + 1) * chunk_fits_) {
            auto ptr = d_ptr + (capacity_ - chunk_fits_) * sizeof(T);
            unmap_chunk(ptr);
            pop_chunk();
        }
    }

    // repeated push/pop calls
    void resize(size_t new_size) {
        if (new_size <= size_) {
            size_t extra = size_ - new_size;
            for(size_t i = 0; i < extra; ++i)
                pop_back();
            return;
        }
        if (new_size > capacity_) reserve(new_size);
        for(size_t i = size_; i < new_size; ++i)
            push_back(T{});
    }

    void reserve(size_t new_capacity) {
        if (new_capacity <= capacity_) return;
        // allocate PA + setup mappings
        size_t new_capacity_ = ROUND_UP(new_capacity, chunk_fits_);
        if (new_capacity_ > max_capacity_) throw std::runtime_error("max_capacity exceeded");

        size_t new_chunks = new_capacity_ / chunk_fits_;
        size_t extra_chunks = new_chunks - capacity_ / chunk_fits_;

        CUdeviceptr ptr = d_ptr + capacity_ * sizeof(T);

        for (size_t i = 0; i < extra_chunks; ++i) {
            auto handle = push_chunk();
            map_chunk(ptr, handle);
            set_access(ptr);
            ptr += chunk_size_;
        }
    }

    size_t size() const { return size_; }
    size_t max_capacity() const { return max_capacity_; }
    size_t capacity() const { return capacity_; }
    size_t chunk_size() const { return chunk_size_; }
    size_t chunk_fits() const { return chunk_fits_; }
    size_t buf_size() const { return size_ * sizeof(T); }
    void set_retain_last(int retain_last) {
        DPRINT("> set_retain_last: %d", retain_last);
        retain_last_ = retain_last;
    }
    int retain_last() const { return retain_last_; }
    T* data() { return (T*)d_ptr; }

    // we are not using UVM
    size_t copy_to_host(T* host_ptr) const {
        size_t sz = size_ * sizeof(T);
        CU_CHECK( cuMemcpyDtoH(host_ptr, d_ptr, sz) );
        return sz;
    }
};