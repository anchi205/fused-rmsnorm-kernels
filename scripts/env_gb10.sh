# Source inside the pytorch-dev-anchita container:  source scripts/env_gb10.sh
# Uses the project venv (which links to /home/dev/work/venv-torch for torch +
# triton) and a CUDA 13 toolkit assembled from pip wheels by
# scripts/setup_cuda13_home.sh. The container's own nvcc is 12.8, which can't
# build for sm_121 or match torch's cu130.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/.venv/bin/activate"
export CUDA_HOME="$ROOT/cuda13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.1"
export MAX_JOBS=16
cd "$ROOT"
