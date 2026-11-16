# torch-vmem: libvmem as PyTorch's CUDA allocator, plus a harness for evict/prefetch policies
# - install() before the first CUDA tensor; backend from VMEM_BACKEND=vmem|uvm|cuda
# - Harness(policy) is a TorchDispatchMode: per op it finds the tensors read and written, calls the policy, and
#   tells the allocator the op used them (last-use event per block). policies decide moves; none is built in
import ctypes
import os
from dataclasses import dataclass, field

import torch
from torch.cuda._sanitizer import FACTORY_FUNCTION_REGEX, ArgumentHandler
from torch.utils._python_dispatch import TorchDispatchMode

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "libtorchvmem.so")
OK, NO_MEMORY, UNSUPPORTED, FATAL = 0, 1, 2, 4
_lib = None


def install(so=SO):
    global _lib
    torch.cuda.memory.change_current_allocator(
        torch.cuda.memory.CUDAPluggableAllocator(so, "vmem_alloc", "vmem_free"))
    _lib = ctypes.CDLL(so)  # same handle as PyTorch's, so same allocator state
    _lib.vmem_touch.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p]
    for f in (_lib.vmem_evict, _lib.vmem_prefetch):
        f.argtypes, f.restype = [ctypes.c_void_p], ctypes.c_int
    _lib.vmem_block.argtypes, _lib.vmem_block.restype = [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_bool


# ---- direct calls: p is a tensor or an address inside one ----

def _addr(p):
    return p.data_ptr() if isinstance(p, torch.Tensor) else p


def evict(p):
    return _lib.vmem_evict(_addr(p))


def prefetch(p):
    return _lib.vmem_prefetch(_addr(p))


# the block holding p: base, size, loc ("device"/"host"), last_op, uses; None if not ours
def block(p):
    out = (ctypes.c_longlong * 5)()
    if not _lib.vmem_block(_addr(p), out):
        return None
    return dict(base=out[0], size=out[1], loc=("device", "host")[out[2]], last_op=out[3], uses=out[4])


# bytes: live tensors, cached, backend on device, backend on host (-1 = unknown)
def info():
    out = (ctypes.c_longlong * 4)()
    _lib.vmem_info(out)
    return dict(zip(("live", "cached", "device", "host"), out))


# ---- harness ----

@dataclass
class Op:
    idx: int        # global op number; also the allocator's last_op
    step: int       # iteration number (Harness.step() bumps it)
    name: str
    reads: set = field(default_factory=set)    # addresses (data_ptr) read only
    writes: set = field(default_factory=set)   # addresses written, outputs included


class Policy:
    # before the op runs: inputs are known, outputs not yet allocated. prefetch here
    def before(self, h, op): pass

    # after the op is queued and touched. evict here (the move waits for this op's kernels)
    def after(self, h, op): pass

    # end of an iteration; h.trace still holds its ops
    def on_step(self, h): pass


class Harness(TorchDispatchMode):
    def __init__(self, policy=None):
        super().__init__()
        self.policy = policy or Policy()
        self.op = 0
        self.steps = 0
        self.trace = []   # ops of the current iteration
        self.prev = []    # ops of the last finished iteration (for look-ahead)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.record_stream.default:
            return func(*args, **kwargs)
        factory = bool(FACTORY_FUNCTION_REGEX.match(func._schema.name))
        ah = ArgumentHandler()
        ah.parse_inputs(func._schema, args, kwargs, is_factory=factory)
        op = Op(self.op, self.steps, func._schema.name, ah.dataptrs_read - ah.dataptrs_written, set(ah.dataptrs_written))
        self.policy.before(self, op)
        out = func(*args, **kwargs)
        ah.parse_outputs(func._schema, out, is_factory=factory)
        op.writes |= ah.dataptrs_written
        ptrs = list(op.reads | op.writes)
        if ptrs:
            arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
            _lib.vmem_touch(arr, len(ptrs), op.idx, torch.cuda.current_stream().cuda_stream)
        self.trace.append(op)
        self.op += 1
        self.policy.after(self, op)
        return out

    # call once per iteration
    def step(self):
        self.policy.on_step(self)
        self.prev, self.trace = self.trace, []
        self.steps += 1
