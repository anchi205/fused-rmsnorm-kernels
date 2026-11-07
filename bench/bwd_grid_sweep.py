"""Sweep the backward persistent-grid size G for both kernels (N=4096, bf16).

    python bench/bwd_grid_sweep.py      -> results/bwd_grid_sweep.csv

Times the raw backward entry points (no autograd), so it isolates the
dw-partials tradeoff: more blocks = more parallelism but more [G, N] fp32
partial traffic.
"""
import os
import sys

import torch
import triton.testing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fused_norm.cuda_ops import cuda_bwd  # noqa: E402
from fused_norm.triton_kernels import triton_bwd  # noqa: E402

N = 4096
rows = []
for M in (16, 128, 512, 2048, 8192):
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dy, ds, s = (torch.randn_like(x) for _ in range(3))
    w = torch.ones(N, device="cuda", dtype=torch.bfloat16)
    rstd = torch.rand(M, device="cuda") + 0.5
    for G in (8, 16, 24, 48, 96, 192, 384):
        os.environ["FUSED_NORM_BWD_GRID"] = str(G)
        for name, f in (("cuda", cuda_bwd), ("triton", triton_bwd)):
            us = triton.testing.do_bench(lambda: f(dy, ds, s, w, rstd), return_mode="median") * 1e3
            rows.append(dict(M=M, N=N, G=G, impl=name, us=us))
    del os.environ["FUSED_NORM_BWD_GRID"]
    for name, f in (("cuda", cuda_bwd), ("triton", triton_bwd)):
        us = triton.testing.do_bench(lambda: f(dy, ds, s, w, rstd), return_mode="median") * 1e3
        rows.append(dict(M=M, N=N, G="heuristic", impl=name, us=us))

import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/bwd_grid_sweep.csv", index=False)
print(df.pivot_table(index=["M", "G"], columns="impl", values="us", sort=False).round(1).to_string())
