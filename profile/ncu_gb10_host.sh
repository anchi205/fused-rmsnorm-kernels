#!/usr/bin/env bash
# Run profile/ncu.sh with GPU counter access on the GB10 box.
#
# Why this exists: the host driver has RmProfilingAdminOnly=1, the dev
# container has no CAP_SYS_ADMIN, and the container's ncu 2025.1.1 can't
# initialize on GB10 ("LibraryNotLoaded"). So this starts a throwaway (--rm)
# container from the same image, with SYS_ADMIN, the host's ncu 2025.3.1
# mounted read-only, and the same work volume.
#
# Run on the host (or from your laptop: ssh $HOST 'bash -s' < profile/ncu_gb10_host.sh).
set -euo pipefail
IMAGE=${IMAGE:-pytorch-dev-anchita}
VOLUME=${VOLUME:-pytorch-anchita-work}
NCU_DIR=${NCU_DIR:-/opt/nvidia/nsight-compute/2025.3.1}
M=${M:-8192}; N=${N:-4096}

docker run --rm --gpus all --cap-add SYS_ADMIN -u 0 \
  -v "$VOLUME":/home/dev/work -v "$NCU_DIR":/opt/ncu:ro \
  -e M="$M" -e N="$N" -e SKIP_FULL="${SKIP_FULL:-}" -e NCU=/opt/ncu/ncu \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch_ext -e TRITON_CACHE_DIR=/tmp/triton \
  "$IMAGE" bash -lc '
    cd /home/dev/work/fused-kernels && source scripts/env_gb10.sh
    bash profile/ncu.sh; rc=$?
    chown -R --reference=. results
    exit $rc'
