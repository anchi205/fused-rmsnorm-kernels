from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_norm",
    version="0.1.0",
    packages=find_packages(include=["fused_norm"]),
    ext_modules=[
        CUDAExtension(
            name="fused_norm_cuda",
            sources=["fused_norm/csrc/binding.cpp", "fused_norm/csrc/fused_add_rmsnorm.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                # -lineinfo lets Nsight Compute map SASS back to .cu source lines.
                "nvcc": ["-O3", "-lineinfo", "--use_fast_math", "-std=c++20"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
