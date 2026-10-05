# Limitations

libvmem (userspace, VMM)
- no hotness info: no access counts, no fault handler to sample; relies on app hints or nothing
- only controls apps that use it, until the v1 hooks; BoxD covers unmodified apps
- no fault handler on VMM ranges, touching unmapped addr crashes kernel (observed, not established)
- handle can't be mapped partially, so move unit = handle, 2 MB min (observed, not established)
- can't beat UVM on raw bandwidth, same PCIe link; loses on unpredictable access
- remap is copy-bound: ~13 GB/s over PCIe; host backing also costs ~89 us/MiB to create (observed, not established)

Nixie
- transparent, so no per-buffer hints, priorities or schedules
- temporal only: one app's working set must fit VRAM
- no quotas or isolation between tenants
- its UVM baseline had no advise/prefetch, so no win shown over tuned UVM
- whole app in or out; victims picked by size, not use
- runtime API only: driver-API and graph-owned memory never moved (inferred from code)
- graph capture guard not wired up (hook misnamed); a switch mid-capture can break it (read in code, not run)

UVM
- device-global eviction, no per-process control
- eviction LRU orders 2 MB chunks by allocation time, not access (driver source)
- evicts then fetches (half-duplex)
- keeps the old CPU page after moving data to GPU, unswappable, until cudaFree; not for pages first touched on GPU (driver source, not run)
