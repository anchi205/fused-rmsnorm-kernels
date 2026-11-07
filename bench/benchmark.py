"""Benchmark fused add+RMSNorm: eager vs torch.compile vs Triton vs CUDA.

    python bench/benchmark.py                 # full sweep -> results/bench.csv
    python bench/benchmark.py --quick         # a few shapes

Timing uses triton.testing.do_bench (CUDA events, L2 flushed before every rep,
median of reps). Bandwidth is *effective* bandwidth: the minimum bytes an
ideal fused kernel must move (fused_norm.reference.bytes_moved) divided by the
time. Using the same numerator for every implementation means eager's extra
memory passes show up as lower GB/s, which is the point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import triton.testing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fused_norm import get_impls  # noqa: E402
from fused_norm.reference import bytes_moved, flops  # noqa: E402

try:  # one compiled graph per shape: lift the recompile limit
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 4096
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 4096
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        torch._dynamo.config.accumulated_recompile_limit = 1 << 20
except Exception:
    pass

EPS = 1e-6
MS = [1, 16, 128, 512, 2048, 8192, 32768]
NS = [1024, 2048, 4096, 5120, 8192]


def bench(fn, grad_to_none=None):
    return triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=grad_to_none,
                                   return_mode="median")


def measure_peaks():
    """Achievable DRAM bandwidth (large D2D copy) and FP32 CUDA-core FLOP/s."""
    n = 1 << 28  # 256M floats = 1 GiB per buffer
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a))
    copy_gbs = 2 * a.numel() * 4 / ms / 1e6
    del a, b
    torch.backends.cuda.matmul.allow_tf32 = False
    k = 8192
    x = torch.randn(k, k, device="cuda")
    y = torch.randn(k, k, device="cuda")
    ms = bench(lambda: x @ y)
    fp32_tflops = 2 * k ** 3 / ms / 1e9
    return {"copy_gbs": copy_gbs, "fp32_tflops": fp32_tflops}


def count_kernels(fn):
    """Number of GPU kernels launched by one call (excluding the warmup)."""
    fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA"
               and "memcpy" not in e.name.lower() and "memset" not in e.name.lower())


def run_shape(name, impl, M, N, dtype, do_bwd=True):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(M, N, device=dev, dtype=dtype)
    r = torch.randn(M, N, device=dev, dtype=dtype)
    w = torch.ones(N, device=dev, dtype=dtype)
    es = x.element_size()
    rows = []

    with torch.no_grad():
        f = lambda: impl(x, r, w, EPS)  # noqa: E731
        ms = bench(f)
        nk = count_kernels(f) if M in (1, 8192) else None
    rows.append(dict(impl=name, pass_="fwd", M=M, N=N, ms=ms, kernels=nk,
                     bytes=bytes_moved(M, N, es, "fwd"), flops=flops(M, N, "fwd")))

    if do_bwd:
        # Only x and w are leaves. If residual were a leaf too, autograd would
        # memcpy the shared dx for the second AccumulateGrad: an extra M*N read
        # and write that isn't part of the op and doesn't happen mid-network.
        # (Found with the profiler; see README.)
        xg, wg = (t.clone().requires_grad_() for t in (x, w))
        rg = r
        y, s = impl(xg, rg, wg, EPS)
        dy, ds = torch.randn_like(y), torch.randn_like(s)
        g = lambda: torch.autograd.backward((y, s), (dy, ds), retain_graph=True)  # noqa: E731
        try:
            ms = bench(g, grad_to_none=[xg, wg])
            nk = count_kernels(g) if M in (1, 8192) else None
            rows.append(dict(impl=name, pass_="bwd", M=M, N=N, ms=ms, kernels=nk,
                             bytes=bytes_moved(M, N, es, "bwd"), flops=flops(M, N, "bwd")))
        except Exception as e:  # e.g. retain_graph unsupported
            print(f"  [{name} bwd M={M} N={N}] skipped: {type(e).__name__}: {e}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    os.makedirs(args.out, exist_ok=True)

    props = torch.cuda.get_device_properties(0)
    peaks = measure_peaks()
    meta = dict(gpu=props.name, sms=props.multi_processor_count,
                l2_mb=getattr(props, "L2_cache_size", 0) / 2 ** 20,
                torch=torch.__version__, triton=__import__("triton").__version__,
                dtype=args.dtype, **peaks)
    print(json.dumps(meta, indent=2))
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    impls = get_impls(include_naive=True)
    Ms, Ns = ([1, 128, 2048, 8192], [2048, 4096, 5120]) if args.quick else (MS, NS)
    rows = []
    for N in Ns:
        for M in Ms:
            for name, fn in impls.items():
                rows += run_shape(name, fn, M, N, dtype, do_bwd=(name != "cuda_naive"))
            done = [r for r in rows if r["M"] == M and r["N"] == N]
            print(f"M={M:>6} N={N:>5}  " + "  ".join(
                f"{r['impl']}/{r['pass_']}={r['ms'] * 1e3:7.1f}us" for r in done), flush=True)
            torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows).rename(columns={"pass_": "pass"})
    df["us"] = df.ms * 1e3
    df["gbs"] = df.bytes / df.ms / 1e6
    df["pct_copy_bw"] = 100 * df.gbs / peaks["copy_gbs"]
    df["gflops"] = df.flops / df.ms / 1e6
    df["ai"] = df.flops / df.bytes
    df.to_csv(os.path.join(args.out, "bench.csv"), index=False)
    print(f"wrote {args.out}/bench.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
