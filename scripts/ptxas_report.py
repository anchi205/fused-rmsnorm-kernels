"""Compile the CUDA kernels standalone and print registers / spills per kernel.

    python scripts/ptxas_report.py [--arch 121]

Spills ("local memory") in a memory-bound kernel are a performance bug: each
spilled value is an extra round-trip through L1/L2 (and possibly DRAM).
"""
import argparse
import os
import re
import subprocess
import sysconfig

from torch.utils.cpp_extension import CUDA_HOME, include_paths

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="121")
ap.add_argument("--src", default="fused_norm/csrc/fused_add_rmsnorm.cu")
args = ap.parse_args()

cmd = [os.path.join(CUDA_HOME, "bin", "nvcc"), "-std=c++20", "-O3", "--use_fast_math",
       f"-arch=sm_{args.arch}", "--expt-relaxed-constexpr", "-Xptxas", "-v",
       "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
       "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
       *[f"-I{p}" for p in include_paths()], f"-I{CUDA_HOME}/include",
       f"-I{sysconfig.get_paths()['include']}", "-c", args.src, "-o", "/tmp/fused_k.o"]
out = subprocess.run(cmd, capture_output=True, text=True)
if out.returncode:
    raise SystemExit(out.stderr[-3000:])

rows, name, spill = set(), None, ("?", "?")
for line in out.stderr.splitlines():
    if m := re.search(r"Compiling entry function '(_Z\w+)'", line):
        name = m.group(1)
    elif m := re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line):
        spill = m.groups()
    elif (m := re.search(r"Used (\d+) registers", line)) and name:
        kern = re.search(r"(fwd_naive_kernel|fwd_kernel|bwd_kernel|dw_reduce_kernel)", name).group(1)
        dt = ("bf16" if "BFloat16" in name else "fp16" if "Half" in name
              else "fp64" if re.search(r"Id[ELi]|IdE", name) else "fp32")
        ints = [int(v) for v in re.findall(r"Li(\d+)E", name)]
        vec, iters = (ints + [None, None])[:2] if kern in ("fwd_kernel", "bwd_kernel") else (None, None)
        dso = "+dso" if kern == "bwd_kernel" and "Lb1E" in name else ""
        tmpl = f"VEC={vec:>2} ITERS={iters:>2} ({vec * iters:>2} elem/thr)" if vec else " " * 30
        rows.add(f"{kern + dso:16s} {dt:5s} {tmpl}  regs={m.group(1):>4}  "
                 f"spill_st/ld={spill[0]}/{spill[1]} B")
        name = None
print("\n".join(sorted(rows)))
