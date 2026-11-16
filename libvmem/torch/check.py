# basic check: a small MLP trains on torch-vmem to the same loss as on the CPU, and a second run leaks nothing
# run: VMEM_BACKEND=vmem|uvm|cuda python check.py
import gc
import os

import torch

import torchvmem

torchvmem.install()


def train(device):
    torch.manual_seed(0)
    # 1024x1024 weights are 4 MiB (large path); biases and the loss are small
    model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 10)).to(device)
    opt = torch.optim.Adam(model.parameters())
    x = torch.randn(256, 1024, generator=torch.Generator().manual_seed(1)).to(device)
    y = torch.randint(0, 10, (256,), generator=torch.Generator().manual_seed(2)).to(device)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()


def held():
    torch.cuda.synchronize()
    gc.collect()
    return torchvmem.info()


backend = os.environ.get("VMEM_BACKEND", "vmem")
print(f"backend = {backend}")
ref, got = train("cpu"), train("cuda")
first = held()  # live here = what PyTorch keeps for the process (cuBLAS workspace)
again = train("cuda")
i = held()
print(f"{backend}: loss cpu {ref:.6f} cuda {got:.6f}, after free {i}")
assert abs(ref - got) < 1e-3 * max(1.0, abs(ref)), "loss differs from CPU"
assert again == got, "second run differs"
assert i["live"] == first["live"], "live bytes grew: leak"
assert i["cached"] > 0, "nothing cached"
print(f"{backend}: ok")
