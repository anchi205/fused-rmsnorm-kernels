#!/usr/bin/env bash
# From the Mac: push the repo into the container and run a command there.
#   export HOST=user@gpu-host   # SSH target running the container
#   scripts/remote.sh sync
#   scripts/remote.sh run "pytest -q tests"
#   scripts/remote.sh pull           # copy results/ back
set -euo pipefail
HOST=${HOST:?set HOST=user@gpu-host}
KEY=${KEY:-~/.ssh/id_ed25519}
CTR=${CTR:-pytorch-dev-anchita}
DEST=/home/dev/work/fused-kernels
cd "$(dirname "$0")/.."
ssh_() { ssh -i "$KEY" -o BatchMode=yes "$HOST" "$@"; }

case "$1" in
  sync)
    COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --exclude results --exclude '__pycache__' --exclude '.pytest_cache' -czf - \
        fused_norm tests bench profile scripts setup.py \
      | ssh_ "docker exec -i $CTR bash -c 'mkdir -p $DEST && tar -xzf - -C $DEST'" ;;
  run)
    shift
    # Command goes over stdin, so no nested quoting is needed.
    printf 'source %s/scripts/env_gb10.sh\n%s\n' "$DEST" "$*" | ssh_ "docker exec -i $CTR bash -l" ;;
  pull)
    mkdir -p results
    ssh_ "docker exec $CTR bash -c 'cd $DEST && tar -czf - results'" | tar -xzf - ;;
  *) echo "usage: $0 sync|run CMD|pull"; exit 1 ;;
esac
