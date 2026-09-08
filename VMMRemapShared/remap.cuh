#pragma once

// cudaremap primitive: swap the physical backing of a virtual range between
// CU_MEM_LOCATION_TYPE_DEVICE and CU_MEM_LOCATION_TYPE_HOST_NUMA *without*
// changing the virtual address, preserving contents.
//
// Plain VMM has no such call. This builds it from the low-level pieces:
//   cuMemCreate (new location) -> stage on a scratch VA -> cuMemcpyDtoD the
//   live bytes -> cuMemUnmap/cuMemRelease the old handle -> cuMemMap the new
//   handle at the ORIGINAL address -> drop the scratch VA.
//
// Single-address-space only. If other processes map the same physical handle
// they must unmap first and re-import afterwards (see owner.cu / subscriber.cu).

#include "common.cuh"

// Move [va, va+size) to host (to_host=true) or back to device.
// `h` is updated in place to the new handle; the old handle is released.
// `acc` is the access descriptor re-applied to the new mapping (a device
// descriptor reaches HOST_NUMA memory over the bus, EGM-style).
// `handle_types` should match what the caller needs to re-export afterwards.
inline void remap_backing(CUdeviceptr va, size_t size,
                          CUmemGenericAllocationHandle& h, bool to_host,
                          const CUmemAccessDesc& acc,
                          unsigned handle_types = 0) {
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = to_host ? CU_MEM_LOCATION_TYPE_HOST_NUMA
                              : CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = 0;  // NUMA node 0 == device 0 on this box
    p.requestedHandleTypes = (CUmemAllocationHandleType)handle_types;

    CUmemGenericAllocationHandle nh;
    CU_CHECK( cuMemCreate(&nh, size, &p, 0) );

    // stage new backing on a scratch VA and copy the current contents in
    CUdeviceptr scratch;
    CU_CHECK( cuMemAddressReserve(&scratch, size, 0, 0, 0) );
    CU_CHECK( cuMemMap(scratch, size, 0, nh, 0) );
    CU_CHECK( cuMemSetAccess(scratch, size, &acc, 1) );
    CU_CHECK( cuMemcpyDtoD(scratch, va, size) );
    CU_CHECK( cuCtxSynchronize() );

    // swap the backing under the original address
    CU_CHECK( cuMemUnmap(va, size) );
    CU_CHECK( cuMemRelease(h) );
    CU_CHECK( cuMemMap(va, size, 0, nh, 0) );
    CU_CHECK( cuMemSetAccess(va, size, &acc, 1) );

    // scratch VA no longer needed (nh stays alive: mapped at va now)
    CU_CHECK( cuMemUnmap(scratch, size) );
    CU_CHECK( cuMemAddressFree(scratch, size) );

    h = nh;
}

// granularity that satisfies both a device and a host allocation of these props
inline size_t remap_granularity() {
    CUmemAllocationProp d = {}, hp = {};
    d.type = hp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    hp.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    size_t g = 0, gh = 0;
    CU_CHECK( cuMemGetAllocationGranularity(&g, &d, CU_MEM_ALLOC_GRANULARITY_MINIMUM) );
    CU_CHECK( cuMemGetAllocationGranularity(&gh, &hp, CU_MEM_ALLOC_GRANULARITY_MINIMUM) );
    return g > gh ? g : gh;
}
