#!/usr/bin/env bash
# Builds a CUDA 13 toolkit dir (./cuda13) from pip wheels: nvcc from
# nvidia-cuda-nvcc, plus runtime/cuBLAS/... headers and libs from the torch
# cu130 install. Run once, from the repo root, inside the container.
set -euo pipefail
TORCH_VENV=${TORCH_VENV:-/home/dev/work/venv-torch}
[ -d .venv ] || uv venv -q .venv --python "$TORCH_VENV/bin/python3"
SP=$(.venv/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
echo "$("$TORCH_VENV/bin/python3" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')" > "$SP/_venv_torch.pth"
uv pip install -q --python .venv/bin/python "nvidia-cuda-nvcc>=13.0,<13.1" "nvidia-cuda-cccl>=13.0,<13.1" "nvidia-cuda-runtime>=13.0,<13.1" "nvidia-cuda-crt>=13.0,<13.1" "nvidia-nvvm>=13.0,<13.1" ninja pytest matplotlib pandas tabulate

A="$SP/nvidia/cu13"
B="$("$TORCH_VENV/bin/python3" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
rm -rf cuda13 && mkdir -p cuda13/include cuda13/lib64
ln -s "$A/bin" cuda13/bin
ln -s "$A/nvvm" cuda13/nvvm
for f in "$A"/include/*; do ln -s "$f" cuda13/include/; done
for f in "$B"/include/*; do [ -e "cuda13/include/$(basename "$f")" ] || ln -s "$f" cuda13/include/; done
for f in "$A"/lib/* "$B"/lib/*; do [ -e "cuda13/lib64/$(basename "$f")" ] || ln -s "$f" cuda13/lib64/; done
ln -sf libcudart.so.13 cuda13/lib64/libcudart.so
cuda13/bin/nvcc --version | tail -1
