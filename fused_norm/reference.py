"""Reference implementations of fused residual-add + RMSNorm.

    s = x + residual                      (the new residual stream, returned)
    y = s * rsqrt(mean(s^2) + eps) * w    (the normalized output, returned)

This is the pre-norm pattern every Llama-style decoder layer runs twice per
layer: ``h = h + attn(...)`` followed by ``rmsnorm(h)``. In eager PyTorch it
is ~7 kernel launches and ~6 full passes over the [M, N] activation tensor.
"""
from __future__ import annotations

import torch


def fused_add_rmsnorm_ref(x: torch.Tensor, residual: torch.Tensor,
                          weight: torch.Tensor, eps: float = 1e-6):
    """Eager PyTorch, written the way Hugging Face's LlamaRMSNorm does it."""
    s = x + residual
    sf = s.float()
    var = sf.pow(2).mean(-1, keepdim=True)
    y = (sf * torch.rsqrt(var + eps)).to(x.dtype) * weight
    return y, s


_compiled = None


def fused_add_rmsnorm_compiled(x, residual, weight, eps: float = 1e-6):
    """Same function through torch.compile (Inductor generates Triton)."""
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(fused_add_rmsnorm_ref, dynamic=False)
    return _compiled(x, residual, weight, eps)


# ---------------------------------------------------------------------------
# Hand-derived backward. Both kernels implement exactly this; tests/test_math.py
# checks it against autograd in float64 on CPU.
#
# Let r = rsqrt(mean(s^2) + eps), g = dy * w. Then
#   dL/ds_i = r * g_i  -  s_i * r^3 * (1/N) * sum_j g_j s_j   (+ ds_out_i)
#   dL/dw   = sum_rows dy * s * r
#   dL/dx = dL/dresidual = dL/ds
# ---------------------------------------------------------------------------
def manual_backward(dy, ds_out, s, weight, rstd):
    """dy, ds_out, s: [M, N]; weight: [N]; rstd: [M] (float32/64)."""
    N = s.shape[-1]
    acc = torch.promote_types(s.dtype, torch.float32)  # fp32, or fp64 if given fp64
    dyf, sf, wf = dy.to(acc), s.to(acc), weight.to(acc)
    r = rstd.to(acc).unsqueeze(-1)
    g = dyf * wf
    dot = (g * sf).sum(-1, keepdim=True)
    ds = r * g - sf * (r ** 3) * dot / N
    if ds_out is not None:
        ds = ds + ds_out.to(acc)
    dw = (dyf * sf * r).sum(0)
    return ds.to(s.dtype), dw.to(weight.dtype)


def bytes_moved(M: int, N: int, elem_size: int, pass_: str) -> int:
    """Minimum DRAM traffic for an ideal *fused* kernel (the roofline numerator).

    fwd: read x, residual, w; write y, s; write rstd (fp32)
    bwd: read dy, ds_out, s, w, rstd; write dx; (dw partials are O(G*N), ignored)
    """
    if pass_ == "fwd":
        return 4 * M * N * elem_size + N * elem_size + 4 * M
    if pass_ == "bwd":
        return 4 * M * N * elem_size + N * elem_size + 4 * M + N * elem_size
    raise ValueError(pass_)


def flops(M: int, N: int, pass_: str) -> int:
    """Rough FP32 FLOP count (adds+muls), for the roofline x-axis."""
    # fwd: add, square, accumulate, scale by rstd, scale by w  -> ~5/elem
    # bwd: g=dy*w, g*s, accumulate, r*g, s*c, sub, +ds_out, dy*s*r (2), dw acc -> ~11/elem
    return (5 if pass_ == "fwd" else 11) * M * N
