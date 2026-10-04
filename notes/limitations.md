# Limitations

libvmem (userspace, VMM)
- no hotness info: no access counts, no fault handler to sample; relies on app hints or nothing
- only controls cooperating processes; BoxD covers unmodified apps
- no fault handler on VMM ranges, touching unmapped addr crashes kernel (observed, not established)
- handle can't be mapped partially, so move unit = handle, 2 MB min (observed, not established)
- can't beat UVM on raw bandwidth, same PCIe link; loses on unpredictable access
- copy was the limit, not remap: ~1.3 ms D2H, ~0.8 ms H2D per 4 MB (observed, not established)

Nixie
- transparent, so no per-buffer hints, priorities or schedules
- temporal only: one app's working set must fit VRAM; no case A, no case D
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
