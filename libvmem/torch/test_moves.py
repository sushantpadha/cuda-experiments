# evict and prefetch by hand: a large tensor goes to host and back with its data intact,
# and the GPU still computes right while it sits on host (kernels read it over PCIe)
# run: VMEM_BACKEND=vmem|uvm|cuda python test_moves.py
import os

import torch

import torchvmem as tv

tv.install()
backend = os.environ.get("VMEM_BACKEND", "vmem")

x = torch.arange(4 << 20, device="cuda", dtype=torch.float32)  # 16 MiB: large, movable
want = x.sum().item()
small = torch.ones(16, device="cuda")                           # 64 B: small slot, never moves
assert tv.evict(small) == tv.UNSUPPORTED, "small tensors must not move"

if backend == "cuda":
    # plain cudaMalloc cannot move
    assert tv.evict(x) == tv.UNSUPPORTED
    print("cuda: moves unsupported, as expected; ok")
    raise SystemExit

assert tv.evict(x) == tv.OK
assert tv.block(x)["loc"] == "host"
assert x.sum().item() == want, "wrong sum while on host"
x.mul_(2)                                                       # write while on host
assert tv.prefetch(x) == tv.OK
assert tv.block(x)["loc"] == "device"
assert x.sum().item() == 2 * want, "data lost in the round trip"
print(f"{backend}: evict, compute on host, prefetch; ok")
