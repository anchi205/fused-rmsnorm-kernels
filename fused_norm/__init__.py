from .reference import fused_add_rmsnorm_ref, fused_add_rmsnorm_compiled  # noqa: F401


def get_impls(include_naive: bool = False):
    """name -> callable(x, residual, weight, eps) -> (y, s). Lazy GPU imports."""
    from .triton_kernels import fused_add_rmsnorm_triton
    from .cuda_ops import fused_add_rmsnorm_cuda, fused_add_rmsnorm_cuda_naive

    impls = {
        "eager": fused_add_rmsnorm_ref,
        "compile": fused_add_rmsnorm_compiled,
        "triton": fused_add_rmsnorm_triton,
        "cuda": fused_add_rmsnorm_cuda,
    }
    if include_naive:
        impls["cuda_naive"] = fused_add_rmsnorm_cuda_naive
    return impls
