# Can VMM remap beat UVM's fault-driven migration?

Status: theory only, 2026-09-24. No measurements. Tracker: 3A, 2Aiii.
Provenance tags as in `01-landscape-survey.md`. Our numbers = preliminary scratch results, RTX 4050 laptop (attribution rule in `CLAUDE.md`).

## Question

Can explicit VMM remap (unmap, copy, map on our schedule) ever beat UVM page-fault migration? Wanted: potentially yes, or never.

## Answer

**Potentially yes, under conditions. Never on raw link bandwidth.** Same bytes, same PCIe link, same ceiling. Remap wins only by (a) fewer bytes, (b) moving earlier, (c) overlap with compute or other link direction. Loses on unpredictable access.

## Where VMM can help: avoiding fault stalls

UVM: GPU access to non-resident page = fault interrupt; warps stall, driver bottom half batches faults, decides eviction/migration, copies, maps, replays. VMM memory mapped and pinned ahead, never faults. Decide residency before kernel: no stall.

- **To check:** fault-handling share of UVM cost (interrupt, batching, replay, zeroing) vs copy. No solid numbers. Search snippet: ~16 to 18 microseconds per GPU fault, CPU-resolved fault unlikely under 10 microseconds. Unverified, secondary. Do not use.
- **Action: consult Soham** (researching page-fault handling): verified per-fault costs, batch behaviour, replayable-fault details, prefetch granularity. Ask before any number enters note or report.
- **Starting papers (unread):** Allen and Ge, "Demystifying GPU UVM cost with deep runtime and workload analysis" (IPDPS 2021, cited by Nixie); Allen, Cooper and Ge, "Fine-grain quantitative analysis of demand paging in unified virtual memory" (TACO 2024, cited by BoxD).

## Other places remap can win

- **Both link directions.** Nixie: UVM evicts then fetches, using half of full-duplex link; ~2x UVM bi-directional throughput with pipelined copies (Nixie Section 2.2, Figure 7, RTX 5090, PCIe 5.0 x8). 4050 laptop link may differ. **[read]**
- **No thrash amplification.** BoxD: 16 GB tenants caused 300 to 475 GB evictions each under global eviction. Explicit residency avoids evicting soon-used data. **[read]**
- **Pinned host memory.** UVM needs pinned CPU page per GPU-resident page (Nixie Section 2.2). VMM puts cold data in host memory, no shadow copy. **[read]**
- **Eviction control.** UVM evicts by own global order. With hints or known access order we pick victim.

## Where remap cannot win

- **Unpredictable or data-dependent access.** Scratch runs: VMM handle not partially mappable (`cuMemMap` needs offset 0, full handle); kernel touching unmapped address crashes. Residency must be right before kernel, at handle granularity (2 MB+). UVM self-corrects at 4 KB to 64 KB. Wrong working-set estimate = crash or large copy, not fault.
- **Small hot sets scattered over large regions.** Coarse handles move too much.
- **Data that fits.** Neither moves anything.
- **Copy-bound cases.** Preliminary: ~1.3 ms device to host, 0.8 ms host to device per 4 MB (~3 and 5 GB/s). Copy was limit, not remap overhead.

## What to measure against

Both baselines required; plain UVM alone overstates benefit.

1. **Plain UVM.** `cudaMallocManaged`, no hints, no prefetch. Shows fault cost, thrash.
2. **UVM with async prefetch and advise.** `cudaMemPrefetchAsync` on stream ahead of use, plus `cudaMemAdvise` (preferred location, accessed-by, read-mostly). Fair baseline: overlaps migration with compute, avoids most first-touch faults. Remap edge narrower: eviction control, no oversubscription thrash, tier choice, two-direction overlap.

Nixie's "UVM" baseline hooked only allocation calls (no advise, no prefetch): its numbers are vs baseline 1, not 2. **[read]**

Also references (from `HANDOFF.md`): `cudaMalloc` with hand-written double buffering where data streams (ideal for case A); `cudaMallocAsync` pools.

Metrics: wall time, stall per kernel, bytes migrated, PCIe utilization per direction, resident set over time, allocator overhead. Tooling: Nsight Systems for UVM fault/migration events (tracker 4A, still to learn). Nsight UVM page-fault tracing reportedly adds up to ~70% overhead: use for timelines, cross-check totals without it.

Workloads: anchor cases A, C, D. Sweep oversubscription ratio. Include a workload where remap should lose (irregular access) to bound claim.

## What we do not have: hotness

Userspace `vmadvise` has no page-hotness info. GPU does not report accesses; VMM ranges have no fault handler to sample. UVM probably has little more:

- Nixie, citing Allen and Ge: UVM LRU metadata updates only on faults. **[read]**
- Secondary source: eviction order = migration recency, not access recency. **[abstract]**
- Hardware access counters exist on Volta+; UVM has access-counter module. Whether they influence eviction on this GPU unverified. Check `uvm_pmm_gpu.c`, `uvm_gpu_access_counters.c` in NVIDIA open kernel modules (driver 580).

**Decision:** no cheap sampling or access-estimation mechanism. Known downside, in `notes/limitations.md`. Residency uses application-declared info (order, priority, lifetime) or nothing.

## Caveat on gate rule

Phase 1 background. Records what we would measure. No API or design proposed. Related-work mechanisms (e.g. Nixie draining kernels before migration) are observations, not choices.
