# v3 harness: the worst possible policy (evict every large tensor after each op, prefetch before each op)
# must still train to the CPU's loss. checks the harness sees forward, backward and optimizer ops,
# and that the allocator keeps per-block state (last op, uses)
# run: VMEM_BACKEND=vmem|uvm|cuda python test_harness.py
import os

import torch

import torchvmem as tv

tv.install()
backend = os.environ.get("VMEM_BACKEND", "vmem")


class EvictAll(tv.Policy):
    def __init__(self):
        self.moves = 0

    def before(self, h, op):
        for p in op.reads | op.writes:
            tv.prefetch(p)   # no-op for a block already on device; unsupported for small ones

    def after(self, h, op):
        for p in op.reads | op.writes:
            b = tv.block(p)
            if b and b["loc"] == "device" and tv.evict(p) == tv.OK:
                self.moves += 1


def train(device, harness=None):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 10)).to(device)
    opt = torch.optim.Adam(model.parameters())
    x = torch.randn(256, 1024, generator=torch.Generator().manual_seed(1)).to(device)
    y = torch.randint(0, 10, (256,), generator=torch.Generator().manual_seed(2)).to(device)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if harness:
            harness.step()
    return loss.item(), model[0].weight


ref, _ = train("cpu")
policy = EvictAll()
h = tv.Harness(policy)
with h:
    got, w = train("cuda", h)
    b = tv.block(w)

names = {op.name for op in h.prev}
print(f"{backend}: loss cpu {ref:.6f} cuda {got:.6f}, {policy.moves} evictions, "
      f"{len(h.prev)} ops in last step, weight block {b}")
assert abs(ref - got) < 1e-3 * max(1.0, abs(ref)), "loss differs from CPU"
assert "aten::addmm" in names and "aten::threshold_backward" in names, "missed forward or backward ops"
assert any("_foreach" in n for n in names), "missed optimizer ops (Adam's foreach kernels)"
assert b["uses"] > 0 and b["last_op"] >= 0, "allocator kept no per-block state"
assert (policy.moves > 0) == (backend != "cuda"), "moves should happen exactly on vmem and uvm"
print(f"{backend}: ok")
