# Research directions after Nixie

2026-09-28. Question: Nixie already does transparent, fault-free GPU multiplexing on VMM. What is left to build? Tags: **[read]** full text or code read earlier (`nixie-summary/`), **[fetched]** paper page or abstract fetched today via small-model summary, **[search]** search snippet only, **[idea]** ours, untested.

## Landscape delta found today

| Work | What it does | Layer | Gap left |
|---|---|---|---|
| Nixie (OSDI '26) **[read]** | whole-app time slicing, VMM remap, 4 tiers, full-duplex copy | userspace shim + daemon | temporal only; immutable weights copied out every switch (paper §8); no hotness; no tuned-UVM baseline |
| MSched, arXiv 2512.24637 (Shen, Chen, Chen, Chen; SJTU) **[fetched]** | predicts each kernel's working set from launch args (NVBit-profiled templates, 0.25% false negatives), Belady eviction over scheduler timeline, pipelined D2H/H2D | **modified GPU driver** (UVM LRU ioctls) + preload DLL + daemon, on XSched | needs driver change; misprediction falls back to UVM page faults; no clean/dirty, dedup, or zero-copy |
| PhoenixOS (SOSP '25) **[fetched]** | speculates kernel read/write sets from args, validates with binary instrumentation; soft copy-on-write for checkpoint | OS-level | checkpoint/restore, not multiplexing |
| Concordia, arXiv 2606.23521 **[search]** | GPU-side diff against device shadow copy, ships only dirty 4 KB pages; 0.04–0.53 ms for 16–256 MB | inference checkpointing | not multiplexing; needs a shadow copy in VRAM |
| Prism (OSDI '26, Yu et al., UCLA/Berkeley; arXiv 2505.04021) **[read]** | multi-LLM serving; VMM "balloon driver" (kvcached) maps 2 MB pages on demand into per-engine reserved VA, so weights and KV cache move between models; spatial and temporal sharing in one scheme; KVPR placement, slack-aware request queue | PyTorch extension (elastic tensors) in SGLang, 22 lines changed; H100 cluster | white-box only; eviction = kill engine, drop memory, reload weights from host DRAM (engine knows weights are clean; KV is discarded, not saved); no dedup; no general CUDA apps |
| vLLM sleep mode **[search]** | offload or discard weights on sleep | inside vLLM | white-box |
| cuda-llm-weight-share, llama.cpp discussion #21223 **[search]** | explicit library sharing one VRAM copy of weights across llama.cpp processes | app-modified | not transparent |
| AutoUVM, arXiv 2609.06172 **[fetched]** | tensor-level prefetch for LLMs on UVM | framework + UVM | UVM only |

Consequence: "hints plus VMM residency" (old `vmadvise` pitch) sits between Nixie (transparent) and Prism/kvcached (white-box), and MSched covers kernel-level prediction. The remaining openings are narrower and more mechanism-specific.

## Candidate directions

### D1. Clean-chunk elision for transparent multiplexing **[idea]**

GPUs have no dirty bit, so Nixie copies every chunk out at every switch, including model weights that never change. Nixie names this as a limit and suggests white-box hints. D1 does it black-box:

1. On H2D load, keep the host copy (in the paged or disk tier, not pinned) and record a hash per 2 MB block.
2. At eviction, hash each block on the GPU. Memory-bound at VRAM bandwidth (RTX 4050 Laptop: about 192 GB/s spec) vs. PCIe (4.0 x8, 16 GB/s spec; our scratch copies 3–5 GB/s).
3. Unchanged block: unmap and release at once, no copy. Changed block: copy as now.

Expected effect: weights (most of an LLM or diffusion footprint) leave VRAM at hash speed; KV cache and activations still copy. VRAM frees earlier, so Nixie's H2D side (gated by "VRAM tokens" from D2H) starts sooner; disk and pinned tiers see fewer writes. Trade: host RAM for the retained clean copy, against Nixie's exactly-one-copy rule.

Honest ceiling: with full-duplex copy, switch time is roughly max(out, in), not the sum. If in ≈ out, eliding "out" helps through earlier VRAM release and less pinned-buffer pressure, not by halving time. Must be measured.

Build: fork Nixie (Rust, open source) or a small C shim reproducing its switch path. One hash kernel, one per-block table. Weeks, not months.

### D2. Content-based sharing and copy-before-write **[idea]**

Two unmodified processes loading the same model (two llama.cpp servers with different settings; ComfyUI workflows sharing a text encoder) hold two VRAM copies. Using D1's hashes, identical blocks map to one physical handle exported across processes (`cuMemExportToShareableHandle`). Result: shared GPU memory like POSIX shm, but found by content and needing no app change.

True copy-on-write is impossible: VMM ranges have no fault handler, so a write to a read-only mapping (`CU_MEM_ACCESS_FLAGS_PROT_READ`) kills the kernel instead of trapping. What works is **copy-before-write**:

1. Before each launch, predict the kernel's write set from its arguments (MSched/PhoenixOS style) or from a hint.
2. If a predicted write hits a shared block, give the writer a private copy, then launch.
3. Keep shared blocks mapped read-only as a tripwire: a missed prediction crashes loudly rather than corrupting the other process.

Weak spot: writes through indirect pointers (pointer arrays, structs in device memory) escape argument scanning. MSched reports under 1% indirect accesses, but for MSched a miss is a page fault; here it is a crash. Measure the miss rate per workload. LLM weights are the easy case: never written.

Order of work: hash infrastructure, D1 clean elision, D2 sharing, then copy-before-write. `madvise`-style hints (read-only, will-need, don't-need) feed the same machinery: a read-only hint makes sharing safe without prediction.

### D3. Concurrent (spatial) multiplexing with a fault-free fallback **[idea]**

Nixie runs one app at a time; MSched runs tasks concurrently but needs a driver change and UVM faults as the safety net. D3 is MSched's idea in userspace on VMM, with a different safety net: an evicted range is never left unmapped. It is remapped to a host-NUMA `cuMemCreate` handle, so a mispredicted access runs as zero-copy over PCIe (slow, correct) instead of crashing. Working sets from launch args, at allocation granularity.

Open risks: unmap-then-map is not atomic, so the buffer must be quiescent during the swap (per-buffer last-use events, but indirect pointers escape tracking); per-launch overhead; zero-copy bandwidth of host-NUMA VMM on this GPU unmeasured. Largest payoff, highest risk for 1.5 months.

### D4. Tuned-UVM vs. Nixie on a low-VRAM laptop GPU (measurement) **[idea]**

Nixie compared only against UVM with no advise or prefetch. Nobody has shown Nixie against UVM with `cudaMemAdvise` + `cudaMemPrefetchAsync`, nor on a 6 GB laptop GPU where oversubscription is the norm. Cheap, needs no new mechanism, fills report section 2.4 (tracker 5A) and produces the baseline numbers D1–D3 need anyway.

### Testing and comparison still missing

Every contrast above is from papers, not runs. Before claiming any gap, run side by side on this machine, same workloads:

- plain UVM, and UVM with `cudaMemAdvise` + `cudaMemPrefetchAsync` (always both);
- Nixie (runnable, open source);
- nvshare; MSched if released; kvcached for LLM-serving cases;
- `cudaMalloc` with manual copies as the hand-tuned bound.

Also read in full: MSched, Concordia, kvcached code. Tracker 4A, 4B, 5D.

### Not pursued

- Hint API alone: white-box hints are Nixie's stated future work, and Prism/kvcached already exploit engine knowledge. Kept only as an input to D1/D2 (a read-only hint makes D2 safe without prediction).
- Nixie engineering gaps (driver-API memory, graph-owned memory, misnamed capture hook): worth an upstream issue, not research.

## Recommendation

Chosen 2026-09-29: D3 is the project (spatial sharing with host fallback, general workloads, optional hints), built in stages (tracker goal 1). D4 stays as the comparison work (tracker goal 5). D1 and D2 are later features (tracker 1F). The 2026-09-28 recommendation (D4, then D1, D2 as stretch) is superseded.

## To verify before committing

- Nixie licence, and whether it runs on 6 GB (it reserves 768 MiB per context plus 256 MiB margins).
- Actual H2D/D2H and duplex bandwidth on this laptop (`nvbandwidth`).
- MSched open-source status and exact driver changes (read full paper).
- Concordia details (only a snippet seen).
- Remapping one VA from a device handle to a host-NUMA handle, and kernel access through it (D3).
