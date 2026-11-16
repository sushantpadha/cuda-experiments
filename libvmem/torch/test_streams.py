# v2: a block freed on one stream and reused on another must wait for the old stream's work
# side stream: fill x with 0, then 1000 slow adds; free x while those are still queued
# main stream: reuse the same memory as y, fill with -1. without the event wait, the adds land in y
# run: VMEM_BACKEND=vmem|uvm|cuda python test_streams.py
import os

import torch

import torchvmem as tv

tv.install()
n = 8 << 20  # 32 MiB of float32
side = torch.cuda.Stream()
with torch.cuda.stream(side):
    x = torch.zeros(n, device="cuda")
    for _ in range(1000):
        x.add_(1)
    old = x.data_ptr()
    del x  # freed on `side` while the adds are still queued

y = torch.empty(n, device="cuda")  # main stream
y.fill_(-1)
torch.cuda.synchronize()
assert y.data_ptr() == old, "block was not reused, test proves nothing"
assert (y == -1).all().item(), "side stream's adds landed in the reused block"
print(f"{os.environ.get('VMEM_BACKEND', 'vmem')}: cross-stream reuse waits; ok")
