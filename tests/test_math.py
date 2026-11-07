"""CPU-only: check the hand-derived backward formula against autograd in fp64."""
import torch

from fused_norm.reference import manual_backward


def test_manual_backward_matches_autograd():
    torch.manual_seed(0)
    M, N, eps = 7, 33, 1e-6
    x = torch.randn(M, N, dtype=torch.float64, requires_grad=True)
    r = torch.randn(M, N, dtype=torch.float64, requires_grad=True)
    w = torch.randn(N, dtype=torch.float64, requires_grad=True)

    s = x + r
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    y = s * rstd * w
    dy, ds = torch.randn_like(y), torch.randn_like(s)
    gx, gr, gw = torch.autograd.grad((y, s), (x, r, w), (dy, ds))

    dx, dw = manual_backward(dy, ds, s.detach(), w.detach(), rstd.squeeze(-1).detach())
    torch.testing.assert_close(dx, gx, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(dx, gr, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(dw, gw, rtol=1e-10, atol=1e-10)
