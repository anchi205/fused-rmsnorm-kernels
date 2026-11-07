"""GPU correctness: Triton and CUDA vs a float64 gold reference, fwd + bwd.

Low-precision kernels can't match fp64 exactly, so each one must be at most
2x as far from gold as eager PyTorch in the same dtype is, plus a small
dtype-dependent slack.
"""
import pytest
import torch

from fused_norm.reference import fused_add_rmsnorm_ref

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SLACK = {torch.float32: 1e-4, torch.float16: 2e-3, torch.bfloat16: 2e-2}
# (M, N). Includes non-power-of-2 widths, widths not divisible by the vector
# size (scalar fallback path), a single row, and M > persistent grid size.
SHAPES = [(1, 4096), (3, 1000), (17, 4100), (128, 768), (257, 5120), (2048, 4096), (64, 8192), (5, 16384)]
EPS = 1e-6


def _impls():
    from fused_norm.cuda_ops import fused_add_rmsnorm_cuda, fused_add_rmsnorm_cuda_naive
    from fused_norm.triton_kernels import fused_add_rmsnorm_triton
    return {"triton": fused_add_rmsnorm_triton, "cuda": fused_add_rmsnorm_cuda,
            "cuda_naive": fused_add_rmsnorm_cuda_naive}


def _run(fn, x, r, w, dy, ds, backward=True):
    x, r, w = (t.detach().clone().requires_grad_(backward) for t in (x, r, w))
    y, s = fn(x, r, w, EPS)
    if not backward:
        return y.detach(), s.detach(), None, None, None
    torch.autograd.backward((y, s), (dy, ds))
    return y.detach(), s.detach(), x.grad, r.grad, w.grad


def _err(a, b):
    return (a.double() - b.double()).abs().max().item()


@cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("M,N", SHAPES)
@pytest.mark.parametrize("impl", ["triton", "cuda", "cuda_naive"])
def test_matches_gold(impl, M, N, dtype):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(M, N, device=dev, dtype=dtype)
    r = torch.randn(M, N, device=dev, dtype=dtype)
    w = (1 + 0.1 * torch.randn(N, device=dev)).to(dtype)
    dy = torch.randn(M, N, device=dev, dtype=dtype)
    ds = torch.randn(M, N, device=dev, dtype=dtype)

    backward = impl != "cuda_naive"
    gold = _run(fused_add_rmsnorm_ref, *(t.double() for t in (x, r, w, dy, ds)))
    eager = _run(fused_add_rmsnorm_ref, x, r, w, dy, ds)
    got = _run(_impls()[impl], x, r, w, dy, ds, backward=backward)

    names = ["y", "s", "dx", "dresidual", "dw"]
    for name, g, e, o in zip(names, gold, eager, got):
        if o is None:
            continue
        assert o.dtype == dtype and o.shape == g.shape, name
        scale = max(1.0, g.abs().max().item())
        bound = 2 * _err(e, g) + SLACK[dtype] * scale
        assert _err(o, g) <= bound, f"{name}: err {_err(o, g):.3e} > bound {bound:.3e}"


@cuda
@pytest.mark.parametrize("impl", ["triton", "cuda"])
def test_only_y_used(impl):
    """If the residual output gets no gradient, backward must still work."""
    torch.manual_seed(0)
    x = torch.randn(8, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
    r = torch.randn_like(x, requires_grad=True)
    w = torch.ones(1024, device="cuda", dtype=torch.float16, requires_grad=True)
    y, _ = _impls()[impl](x, r, w, EPS)
    y.float().sum().backward()

    x2, r2, w2 = (t.detach().clone().requires_grad_() for t in (x, r, w))
    y2, _ = fused_add_rmsnorm_ref(x2, r2, w2, EPS)
    y2.float().sum().backward()
    torch.testing.assert_close(x.grad, x2.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w.grad, w2.grad, atol=2e-1, rtol=2e-2)


@cuda
def test_3d_input_shape():
    from fused_norm.triton_kernels import fused_add_rmsnorm_triton
    from fused_norm.cuda_ops import fused_add_rmsnorm_cuda
    x = torch.randn(2, 5, 2048, device="cuda", dtype=torch.bfloat16)
    r, w = torch.randn_like(x), torch.ones(2048, device="cuda", dtype=torch.bfloat16)
    for fn in (fused_add_rmsnorm_triton, fused_add_rmsnorm_cuda):
        y, s = fn(x, r, w, EPS)
        assert y.shape == x.shape and s.shape == x.shape
