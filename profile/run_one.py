"""Minimal driver for Nsight Compute: runs one impl/pass/shape a few times.

    ncu ... python profile/run_one.py --impl cuda --pass fwd --M 8192 --N 4096
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fused_norm import get_impls  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--impl", default="cuda")
ap.add_argument("--pass", dest="pass_", default="fwd", choices=["fwd", "bwd"])
ap.add_argument("--M", type=int, default=8192)
ap.add_argument("--N", type=int, default=4096)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--iters", type=int, default=3)
a = ap.parse_args()

fn = get_impls(include_naive=True)[a.impl]
dt = getattr(torch, a.dtype)
x = torch.randn(a.M, a.N, device="cuda", dtype=dt, requires_grad=a.pass_ == "bwd")
r = torch.randn_like(x)  # not a leaf: avoids the AccumulateGrad dx copy (see README)
w = torch.ones(a.N, device="cuda", dtype=dt, requires_grad=a.pass_ == "bwd")

if a.pass_ == "fwd":
    with torch.no_grad():
        for _ in range(a.iters):
            fn(x, r, w, 1e-6)
else:
    y, s = fn(x, r, w, 1e-6)
    dy, ds = torch.randn_like(y), torch.randn_like(s)
    torch.cuda.synchronize()
    for _ in range(a.iters):
        torch.autograd.backward((y, s), (dy, ds), retain_graph=True)
        x.grad = w.grad = None
torch.cuda.synchronize()
