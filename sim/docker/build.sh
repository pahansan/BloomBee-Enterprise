#!/usr/bin/env bash
# Build the self-contained simulation image.
#
#   ./sim/docker/build.sh cpu     # default
#   ./sim/docker/build.sh cuda    # host with NVIDIA Container Toolkit
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-${MODE:-cpu}}"

case "$MODE" in
  cpu)
    IMAGE_TAG="${IMAGE_TAG:-bloombee-sim:cpu}"
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
    TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0+cpu}"
    ;;
  cuda|gpu)
    IMAGE_TAG="${IMAGE_TAG:-bloombee-sim:cuda}"
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
    TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0+cu128}"
    ;;
  *)
    echo "usage: $0 [cpu|cuda]" >&2
    exit 2
    ;;
esac

echo "[build] image=${IMAGE_TAG} torch=${TORCH_SPEC}"
docker build \
  -f "$ROOT/sim/docker/Dockerfile" \
  --build-arg "TORCH_INDEX=${TORCH_INDEX}" \
  --build-arg "TORCH_SPEC=${TORCH_SPEC}" \
  -t "$IMAGE_TAG" \
  "$ROOT"
echo "[build] done: ${IMAGE_TAG}"
