# Benchmarks ideas

all TBD, nothing run

workloads
- batch image/video with naive kernels, dataset > VRAM (case A)
- multi-process jobs on one GPU, compare with BoxD's 3-process setup (case C)
- growing buffers, append/resize, KV-cache-like (case D)
- rodinia: hotspot, SRAD (regular), BFS (irregular, UVM should thrash)
- UVM vs VMM microbench on oversubscribed working set (3A)
- control runs that should not benefit: data fits in VRAM

alternatives
- plain UVM and UVM + advise/prefetch, always both (cuda toolkit)
- cudaMalloc + manual double buffering, cudaMallocAsync pools (cuda toolkit)
- Nixie https://arxiv.org/abs/2601.11743 (paper), code https://github.com/XOR-op/Nixie (OSDI '26), 5090 vs our 4050
- TGS (Nixie ref [34])
- nvshare (Nixie ref [3]) repo https://github.com/grgalex/nvshare
- BoxD, local pdf in references/ (prof)
- GMLake https://github.com/antgroup/glake (repo), case D
- MSched https://arxiv.org/abs/2512.24637 (modified driver, maybe no code yet)
- kvcached https://github.com/ovg-project/kvcached (LLM serving on VMM)

note: all contrast so far is from papers. need real side-by-side runs here (tracker 3D)
