"""PyTorch binding for the hand-written CUDA kernels.

Uses the prebuilt extension if `pip install -e .` was run. Otherwise the
extension is JIT-compiled on first use with torch.utils.cpp_extension.load.
"""
from __future__ import annotations

import os

import torch

_ext = None


def _load():
    global _ext
    if _ext is not None:
        return _ext
    try:
        import fused_norm_cuda as ext  # built by setup.py
    except ImportError:
        from torch.utils.cpp_extension import load
        here = os.path.join(os.path.dirname(__file__), "csrc")
        ext = load(
            name="fused_norm_cuda",
            sources=[os.path.join(here, "binding.cpp"), os.path.join(here, "fused_add_rmsnorm.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math", "-std=c++20"],
            extra_cflags=["-O3", "-std=c++20"],
            verbose=bool(os.environ.get("FUSED_NORM_VERBOSE")),
        )
    _ext = ext
    return ext


def cuda_fwd(x, residual, weight, eps, naive=False):
    return _load().fwd(x.contiguous(), residual.contiguous(), weight.contiguous(), eps, naive)


def cuda_bwd(dy, ds_out, s, weight, rstd):
    ds_out = ds_out.contiguous() if ds_out is not None else None
    return _load().bwd(dy.contiguous(), ds_out, s, weight, rstd)


class _CudaFusedAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        y, s, rstd = cuda_fwd(x, residual, weight, eps)
        ctx.save_for_backward(s, weight, rstd)
        ctx.set_materialize_grads(False)
        return y, s

    @staticmethod
    def backward(ctx, dy, ds):
        s, weight, rstd = ctx.saved_tensors
        if dy is None:
            dy = torch.zeros_like(s)
        dx, dw = cuda_bwd(dy, ds, s, weight, rstd)
        return dx, dx, dw, None


def fused_add_rmsnorm_cuda(x, residual, weight, eps: float = 1e-6):
    return _CudaFusedAddRMSNorm.apply(x, residual, weight, eps)


def fused_add_rmsnorm_cuda_naive(x, residual, weight, eps: float = 1e-6):
    """Forward only: the unoptimized baseline kernel."""
    y, s, _ = cuda_fwd(x, residual, weight, eps, naive=True)
    return y, s
