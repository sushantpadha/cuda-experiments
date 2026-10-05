# Explainer: "Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management"

Bakita and Anderson, UNC Chapel Hill, RTAS 2024. Local copy: `references/Demystifying_NVIDIA_GPU_Internals_to_Enable_Reliable_GPU_Management.pdf` (12 pages). Everything below restates the paper unless a line is marked **[ours]**.

## What the paper is about, in one paragraph

Real-time systems (self-driving cars) need guarantees on how long GPU work takes. Three earlier GPU-management schemes gave guarantees that were wrong, because each assumed something false about how NVIDIA GPUs schedule work. The authors dug into the hardware (open-source drivers, NVIDIA patents, register docs, and their own kernel module `nvdebug`) and wrote down 8 rules (R1 to R8) for how work really flows from a CUDA call to a GPU engine. Then they showed each broken scheme fails because it ignores one of these rules.

The paper is about **scheduling**: who gets which part of the GPU, and when. It says almost nothing about memory management or VMM (see the last section).

## The parts of a GPU (Sec. II)

A GPU is not one accelerator. It is a set of independent units called **engines**, all connected to GPU memory through an internal crossbar. The crossbar also reaches PCIe, so engines can access CPU memory, slowly.

| Engine | Does |
|---|---|
| Compute/Graphics engine | all the general-purpose cores (the SMs). Runs kernels |
| Copy engines | move data between GPU memory, CPU memory, other GPUs, asynchronously. An A100 (GA100) has five |
| Video encode / decode, JPEG decode, optical flow | special-purpose media work |

## The software words (Sec. II)

- **Context**: one GPU virtual address space per GPU-using program ("task"). All of a program's allocations, copies and kernels live in it. A program can make several, but that is discouraged; the paper assumes one per program.
- **Stream**: a FIFO queue of GPU operations (kernel launches, copies) inside a context. If you don't name one, CUDA uses a default stream. Calls with a CPU-visible result (like copying results back) block until earlier stream work finishes.
- **Kernel**: a function run on the compute engine, as many parallel blocks of threads.

## How a kernel launch reaches the hardware (Sec. IV, Fig. 3 and 4)

```
 your program
   │  kernel<<<...>>>(...)  /  cudaMemcpyAsync(...)
   ▼
 CUDA runtime lib ─► CUDA driver lib          (userspace)
   │   writes commands straight into a buffer the GPU can read: no system call
   ▼
 ┌──────────────── one context ────────────────┐
 │ TSG (time-slice group)                      │
 │   channel ── pushbuffer (command queue)     │  ◄─ each stream is mapped onto a channel
 │   channel ── pushbuffer                     │
 │   ... (8 compute channels by default)       │
 └─────────────────────────────────────────────┘
   │  the TSG sits on a runlist
   ▼
 runlist  ─►  Host Interface (the hardware scheduler)
   │            1. round-robin time slices across TSGs (= across contexts)
   │            2. inside the active TSG, scan its channels for pending commands
   ▼
 PBDMA (pushbuffer DMA unit): pulls the commands into the GPU and parses them
   ▼
 engine (compute, copy, ...) runs the work
```

Step by step, in the paper's numbering:

1. **Setup.** When a program starts using the GPU, the kernel driver creates **pushbuffers** (command queues). Each pushbuffer plus bookkeeping is a **channel**. All of a program's channels plus its context information form one **Time-Slice Group (TSG)**. The TSG is put on a **runlist**, and the program gets a pointer to its pushbuffers.
2. **Streams to channels.** Your stream operations become commands written into a channel's pushbuffer. Because the pushbuffer is mapped into the program's memory, launching work needs no system call. (This is also why launches are hard to monitor from outside.)
3. **Channels to runlists.** The hardware scheduler (NVIDIA calls it the **Host Interface**) chooses what runs in two levels: it time-slices between TSGs round-robin, and within the active TSG it cycles through the channels looking for pending commands.
4. **Runlists to engines.** For the chosen channel, a **PBDMA** unit fetches the commands and hands them to the right engine.

So, mapped to everyday CUDA words:

| You write | The GPU sees |
|---|---|
| a process using the GPU (one context) | one TSG |
| a stream | a channel (one of a fixed pool) |
| a kernel launch / copy | commands in that channel's pushbuffer |
| "the GPU switches between programs" | the scheduler time-slicing TSGs on a runlist |

## The 8 rules (Sec. V), each with the experiment behind it

Tested on 9 GPUs from 2016 to 2022 (Pascal to Ada, desktop and Jetson).

**Channels**

- **R1. Every operation that uses an engine goes through a channel.** They disabled all of a program's channels with `nvdebug`: kernel launches, copies, and even device-mapped memory allocations all stalled until the channels were re-enabled.
- **R2. The number of channels limits parallelism inside a program.** CUDA creates only **8 compute channels per context** by default (x86, through at least CUDA 12.2; as few as 2 on Jetson boards). With 9 streams, the 9th has no channel and must wait for another stream's work to finish dispatching, even though GPU cores are idle: a *false dependency* (Fig. 5). Raising `CUDA_DEVICE_MAX_CONNECTIONS` gives 9 channels and the problem disappears. Two side facts:
  - a channel frees up only after *all* blocks of its stream's last kernel have been dispatched;
  - streams waiting for a channel are **not** served first-come-first-served, so later work can overtake earlier work.

**Runlists**

- **R3. A channel must be on a runlist to be scheduled.** Every enabled channel they observed belonged to a runlist.
- **R4. A runlist runs at most one program at a time per engine it serves.** Two programs that each need only a fraction of the GPU still run in mutually exclusive time slices, about **2 ms** each for compute on a GTX 1060 (Fig. 6); two copy-only programs trade off about every **1 ms** (Fig. 7). Creating a new CUDA context also interrupts other programs' compute for about **100 ms**.
- **R5. The number of runlists limits independent parallelism between programs.** Compute and copy normally sit on separate runlists, so one program's copies and another's kernels really overlap (Fig. 8). On the Jetson TX2, compute and copy share one runlist, so a copy-only program gets time-sliced against a compute-only one even though they use different engines (Fig. 9).

**Runlists to engines**

- **R6. A runlist can serve more than one engine.** Every GPU tested has a runlist that handles both compute and copy. In the topology tables, Runlist 0 serves the compute/graphics engine together with the two graphics copy engines, and every other runlist serves exactly one engine. A recent GPU (RTX 6000 Ada) has 17 runlists (Table IV), so lots of room for independent work.
- **R7. Each engine belongs to exactly one runlist.** This is fixed in hardware (the device-topology registers map each engine to one runlist).
- **R8. Copy engines can still interfere with each other.** The copy engines CUDA reports are *logical* copy engines (LCEs). The copying is actually done by *physical* copy engines (PCEs), and registers decide which LCE uses which PCE. The two graphics copy engines (GRCEs) can share a PCE with another LCE. On the RTX 6000 Ada, an OpenGL texture upload slowed an unrelated CUDA GPU-to-CPU copy about **2×** because both landed on the same PCE (Fig. 10, 11). The older GTX 1080 Ti did not have this sharing.

## Why the old schemes broke (Sec. VI)

| Scheme | What it assumed | Rule it broke | What goes wrong |
|---|---|---|---|
| Yang et al.: analysis with no management, one stream per job | kernels run in launch order | R2 | with more than 8 streams, job 10 can get a channel before job 9 and run first, so the timing bound is wrong |
| Capodieci et al.: preemptive EDF by rewriting the runlist | one runlist, one active program | R4, R5, R7 | on Jetson Xavier, preempting a program on the compute runlist leaves its copies running on the copy runlist, delaying the higher-priority program |
| Elliott et al.: one lock per copy engine and for compute | copy engines are independent | R6, R8 | two "different" copy engines share one physical engine, so copies take twice as long as the locks promise |

## nvdebug (Appendix A)

A Linux kernel module that reads GPU registers directly, bypassing the NVIDIA driver. It exposes files under `/proc/gpuX/`:

| File | Shows |
|---|---|
| `device_info` | engines and which runlist each is on |
| `runlistY` | the TSGs and channels currently on runlist Y (time-slice settings, enabled, busy, faulted) |
| `disable_channel`, `enable_channel` | turn a channel off or on |
| `lce_for_pceY`, `shared_lce_for_grceY`, `pce_map` | the logical-to-physical copy engine wiring |

Code and the benchmarks (`exec_logger`, `copy_monitor`) are at <https://www.cs.unc.edu/~jbakita/rtas24-ae/>.

## What the paper says about MPS (footnote 13)

Since Volta, under MPS each application runs as a **subcontext** of one MPS-created context. Rule R4 then does not apply per application. The authors guess the rules still hold if you treat all MPS clients together as one program, but did not verify it.

## What the paper does not cover: VMM

The paper never discusses CUDA's Virtual Memory Management API (`cuMemCreate`, `cuMemMap`, ...), page tables, or how allocations are implemented. The only memory facts it gives:

- a context *is* the program's GPU virtual address space;
- engines reach CPU memory over PCIe through the crossbar, "optionally, but slowly";
- device-mapped allocations go through channels too (R1), so a program with disabled channels cannot allocate.

For VMM itself, the CUDA Programming Guide's Virtual Memory Management chapter is the source.

## Connecting it to what we measured **[ours]**

- Our 2A result (two processes, each kernel takes 2× longer together without MPS) is R4 in action: each process is one TSG on the compute runlist, and only one TSG runs at a time.
- Under MPS the clients become subcontexts of one context (footnote 13), which is consistent with them running concurrently and with one client's fault taking down the others (2A).
- The RTX 4050 here is Ada (compute capability 8.9), the same family as the paper's RTX 6000 Ada, so the copy-engine sharing in R8 could apply. Not checked; `nvdebug` would show it.
- R2 matters for later prototypes: more than 8 streams in one process causes false dependencies unless `CUDA_DEVICE_MAX_CONNECTIONS` is raised.

## Glossary

| Term | Meaning |
|---|---|
| Engine | an independent GPU unit: compute/graphics, copy, video, JPEG |
| SM | streaming multiprocessor, a group of cores inside the compute engine |
| Context | a program's GPU virtual address space and state |
| Stream | FIFO queue of GPU work in a context |
| Pushbuffer | the in-memory command queue the GPU reads |
| Channel | a pushbuffer plus bookkeeping; what a stream runs on |
| TSG | time-slice group: all of one program's channels plus its context info; the unit the scheduler time-slices |
| Runlist | a hardware list of schedulable TSGs/channels, bound to one or more engines |
| Host Interface | NVIDIA's name for the hardware scheduler that picks channels from runlists |
| PBDMA | pushbuffer DMA unit: fetches and parses commands for an engine |
| LCE / GRCE / PCE | logical copy engine (what CUDA sees) / graphics copy engine (LCE 0 and 1) / physical copy engine (does the copy) |
| False dependency | a stream waiting on an unrelated stream because there are not enough channels |
