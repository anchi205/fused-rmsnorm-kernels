"""Fused residual-add + RMSNorm in Triton: forward and backward.

Forward: one program per row. The whole row is loaded into registers once,
s = x + r is written out (the new residual stream), the sum of squares is
reduced in-register, and y is written. 4 passes of DRAM traffic instead of
eager's ~12.

Backward: a persistent grid of G programs (about 1 per SM; see bwd_grid). Each program walks
rows pid, pid+G, ... and computes dx in a single pass. It keeps a float32 dw
accumulator in registers across all of its rows, then writes one [N] partial.
The [G, N] partials are summed afterwards, which is tiny next to [M, N].
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(X, R, W, Y, S, RSTD, stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    off = row * stride + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + off, mask=mask, other=0.0).to(tl.float32)
    # Round s to the storage dtype before using it, so the result matches eager
    # PyTorch exactly. Eager computes s = x + r in the input dtype.
    s_lp = (x + r).to(S.dtype.element_ty)
    tl.store(S + off, s_lp, mask=mask)
    s = s_lp.to(tl.float32)

    var = tl.sum(s * s, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    tl.store(RSTD + row, rstd)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    # Match LlamaRMSNorm: round the normalized value to the input dtype, then scale.
    y = (s * rstd).to(Y.dtype.element_ty).to(tl.float32) * w
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(DY, DSO, S, W, RSTD, DX, DWP, stride, M, N,
                HAS_DSO: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for row in range(pid, M, G):
        off = row.to(tl.int64) * stride + cols
        dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S + off, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row)

        g = dy * w
        dot = tl.sum(g * s, axis=0)
        dx = rstd * g - s * (rstd * rstd * rstd * dot / N)
        if HAS_DSO:
            dx += tl.load(DSO + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=mask)
        dw_acc += dy * s * rstd

    tl.store(DWP + pid.to(tl.int64) * N + cols, dw_acc, mask=mask)


def bwd_grid(M: int, num_sms: int) -> int:
    """Persistent-grid size for the backward (same rule as the CUDA kernel).

    Each block writes an [N] fp32 dw partial that must be re-read, so more
    blocks cost 8*N bytes each. The sweep in bench/bwd_grid_sweep.py on GB10
    (48 SMs) shows the optimum is about one block per SM, at least ~8 rows
    per block, and at least 16 blocks. 4*SMs (the first version) was up to
    1.4x slower at M=128.
    """
    if os.environ.get("FUSED_NORM_BWD_GRID"):  # override for sweeps
        return max(1, min(M, int(os.environ["FUSED_NORM_BWD_GRID"])))
    return max(1, min(M, num_sms, max(16, -(-M // 8))))


def _launch_cfg(N: int, elem_size: int):
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N * elem_size > 65536:
        raise ValueError(f"hidden size {N} too large for single-pass row kernel")
    # About 1 warp per 256 elements, clamped to [1, 16] warps. Wider rows get
    # more warps so each thread keeps ~8 elements in registers.
    num_warps = min(max(BLOCK_N // 256, 1), 16)
    return BLOCK_N, num_warps


def triton_fwd(x, residual, weight, eps):
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    r2 = residual.reshape(-1, N).contiguous()
    M = x2.shape[0]
    y = torch.empty_like(x2)
    s = torch.empty_like(x2)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_N, num_warps = _launch_cfg(N, x.element_size())
    _fwd_kernel[(M,)](x2, r2, weight, y, s, rstd, x2.stride(0), N, eps,
                      BLOCK_N=BLOCK_N, num_warps=num_warps)
    return y.view(shape), s.view(shape), rstd


def triton_bwd(dy, ds_out, s, weight, rstd):
    shape = s.shape
    N = shape[-1]
    s2 = s.reshape(-1, N)
    M = s2.shape[0]
    dy2 = dy.reshape(-1, N).contiguous()
    dso2 = ds_out.reshape(-1, N).contiguous() if ds_out is not None else None
    dx = torch.empty_like(s2)
    BLOCK_N, num_warps = _launch_cfg(N, s.element_size())
    sm = torch.cuda.get_device_properties(s.device).multi_processor_count
    G = bwd_grid(M, sm)
    dwp = torch.empty((G, N), dtype=torch.float32, device=s.device)
    _bwd_kernel[(G,)](dy2, dso2 if dso2 is not None else dy2, s2, weight, rstd,
                      dx, dwp, s2.stride(0), M, N,
                      HAS_DSO=dso2 is not None, BLOCK_N=BLOCK_N, num_warps=num_warps)
    dw = dwp.sum(0).to(weight.dtype)
    return dx.view(shape), dw


class _TritonFusedAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        y, s, rstd = triton_fwd(x, residual, weight, eps)
        ctx.save_for_backward(s, weight, rstd)
        ctx.set_materialize_grads(False)
        return y, s

    @staticmethod
    def backward(ctx, dy, ds):
        s, weight, rstd = ctx.saved_tensors
        if dy is None:
            dy = torch.zeros_like(s)
        dx, dw = triton_bwd(dy, ds, s, weight, rstd)
        # x and residual get the same gradient, since s = x + residual.
        return dx, dx, dw, None


def fused_add_rmsnorm_triton(x, residual, weight, eps: float = 1e-6):
    return _TritonFusedAddRMSNorm.apply(x, residual, weight, eps)
