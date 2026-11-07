"""List GPU kernels (with device time) for one fwd or bwd call of an impl.

    python profile/list_kernels.py --impl compile --pass bwd --M 8192 --N 8192

Uses the PyTorch profiler (CUPTI tracing), which needs no admin rights. As in
the benchmark, only x and w are autograd leaves.
"""
import argparse
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fused_norm import get_impls  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--impl", default="compile")
ap.add_argument("--pass", dest="pass_", default="bwd")
ap.add_argument("--M", type=int, default=8192)
ap.add_argument("--N", type=int, default=8192)
a = ap.parse_args()

fn = get_impls(include_naive=True)[a.impl]
x = torch.randn(a.M, a.N, device="cuda", dtype=torch.bfloat16, requires_grad=a.pass_ == "bwd")
r = torch.randn(a.M, a.N, device="cuda", dtype=torch.bfloat16)
w = torch.ones(a.N, device="cuda", dtype=torch.bfloat16, requires_grad=a.pass_ == "bwd")
y, s = fn(x, r, w, 1e-6)
dy, ds = torch.randn_like(y), torch.randn_like(s)
step = (lambda: torch.autograd.backward((y, s), (dy, ds), retain_graph=True)) if a.pass_ == "bwd" \
    else (lambda: fn(x, r, w, 1e-6))
for _ in range(3):
    step(); x.grad = w.grad = None
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    step(); torch.cuda.synchronize()
tot = 0.0
for e in p.events():
    if e.device_type.name == "CUDA":
        tot += e.device_time
        print(f"{e.device_time:9.1f} us  {e.name[:90]}")
print(f"{tot:9.1f} us  total")
