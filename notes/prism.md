# Prism

> AI-generated :)

[paper (OSDI '26)](https://www.usenix.org/conference/osdi26/presentation/yu-shan) | [arxiv](https://arxiv.org/abs/2505.04021) | [code (kvcached)](https://github.com/ovg-project/kvcached) | local: `references/prism-paper.pdf`

- focused on datacenter LLM serving: many models, most idle most of the time, traffic bursty
- neither pure time sharing (thrashes when two models are busy together) nor fixed partitions (idle model hogs memory) works on real traces

- idea: memory ballooning, borrowed from VMware ESX
    - each serving engine reserves a big VA range, physical 2 MB pages mapped only when needed (VMM)
    - each model gets a memory limit; lower it (inflate) and the engine gives back free KV pages, raise it (deflate) and it can grow
    - idle model: kill its engine, drop its memory; weights reload later from host RAM
    - one mechanism gives both spatial and temporal sharing
- plus: places models across GPUs by KV memory pressure, orders requests on each GPU by deadline slack

- benefits:
    - no copy on reclaim: weights are clean (host copy exists), KV is just freed
    - engine picks what to free, so it never loses live data
    - small integration: PyTorch extension, 22 lines changed in SGLang, attention kernels and CUDA graphs untouched
    - up to 3.3x better TTFT SLO attainment, 2x cost cut; in production at 10K+ GPUs

- lacks:
    - white-box: only works inside an LLM serving engine, not general CUDA apps
    - shrinking is slow, waits for in-flight requests to finish (seconds)
    - no weight sharing across models
    - fast loading relies on NVLink and many GPUs, nothing for a single laptop GPU

- for us: shows cheap reclaim needs knowing what's clean. hashing gets that without the engine; madvise-style hints get it by asking

## Glossary

- **serving engine**: program that keeps one LLM on the GPU and answers requests (SGLang, vLLM, llama-server)
- **tenant**: one of several users sharing the GPU (a process or model)
- **central manager**: the one piece that sees all tenants and decides who gets memory (Prism: kvcached + scheduler; Nixie: daemon; ESX: hypervisor)
- **hypervisor / guest OS**: software running VMs / the OS inside a VM
- **balloon**: driver inside a tenant, controlled by the manager, that holds memory on the manager's behalf
- **inflate / deflate**: grow the balloon so the tenant has less memory / shrink it so the tenant gets memory back
- **reclaim**: take memory back from a tenant, by force (Nixie copies out) or by asking (balloon)
- **request**: one prompt sent to the model
- **token**: roughly one word, what the LLM generates
- **KV cache**: per-request attention state, grows with length, freed when the request ends; always dirty
- **weights**: model parameters, loaded once, only read; always clean
- **clean / dirty**: unchanged since load, so a copy elsewhere is valid / changed, must copy out before freeing
- **in-flight request**: still generating; its KV cache can't be freed yet
- **SLO**: latency target, e.g. first token within 1 s; attainment = % of requests that met it
- **TTFT / TPOT**: wait until first word / gap between words while streaming
- **spatial / temporal sharing**: tenants use the GPU at the same time with part of memory each / take turns with all of it
- **bursty**: traffic in sudden spikes with idle gaps
- **placement**: which GPU each model runs on
- **engine pool**: engines started ahead of time so a model switch skips startup
- **reserve VA / map / unmap**: claim addresses with no memory / attach or detach physical memory
- **page (here)**: 2 MB chunk of physical GPU memory, smallest VMM unit on this hardware
