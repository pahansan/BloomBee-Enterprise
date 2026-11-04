#!/usr/bin/env bash
# Build the BloomBee worker image.
#   ./build.sh cpu            # ~1.9 GB, runs anywhere
#   ./build.sh cu128          # ~7.7 GB on disk (~4.2 GB pushed)
#   ./build.sh cu126          # fallback for older drivers (torch 2.7.1, same kernels)
# Extra args are forwarded to `docker build`.
set -euo pipefail

MODE="${1:-cpu}"
shift || true

case "$MODE" in
  cpu)   TORCH_INDEX="https://download.pytorch.org/whl/cpu";   TORCH_SPEC="torch==2.11.0+cpu" ;;
  cu128) TORCH_INDEX="https://download.pytorch.org/whl/cu128"; TORCH_SPEC="torch==2.11.0+cu128" ;;
  cu126) TORCH_INDEX="https://download.pytorch.org/whl/cu126"; TORCH_SPEC="torch==2.7.1+cu126" ;;
  *) echo "usage: $0 {cpu|cu128|cu126} [docker build args]" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GIT_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/src/bloombee/__init__.py" | head -1)"
VERSION="${VERSION:-dev}"
IMAGE="${IMAGE:-bloombee:${VERSION}-${MODE}}"
# TAG_SHA=1 appends -g<git sha>: a human-readable, immutable-by-convention tag.
[[ "${TAG_SHA:-0}" == "1" ]] && IMAGE="${IMAGE}-g${GIT_SHA}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "image: $IMAGE"; echo "torch: $TORCH_SPEC"; echo "revision: $GIT_SHA"; exit 0
fi

DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f "$ROOT/deploy/docker/Dockerfile" \
  --build-arg TORCH_INDEX="$TORCH_INDEX" \
  --build-arg TORCH_SPEC="$TORCH_SPEC" \
  --build-arg GIT_SHA="$GIT_SHA" \
  --build-arg BLOOMBEE_VERSION="$VERSION" \
  -t "$IMAGE" "$@" "$ROOT"

docker image inspect -f "image {{.Id}}  revision=${GIT_SHA}  size={{.Size}}" "$IMAGE"
