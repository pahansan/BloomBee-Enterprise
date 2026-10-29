#!/usr/bin/env bash
# Diagnose NVIDIA Container Toolkit + Docker GPU access.
#
# Exit 0 only if a container can actually see the GPU.
set -u

echo "=== host GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo
echo "=== toolkit ==="
command -v nvidia-ctk || true
nvidia-ctk --version 2>&1 | head -2 || true
command -v nvidia-container-cli || true
nvidia-container-cli --version 2>&1 | head -2 || true

echo
echo "=== CDI devices ==="
nvidia-ctk cdi list 2>&1 | head -10 || true

echo
echo "=== Docker context ==="
docker context show 2>&1 || true
docker info 2>&1 | grep -E 'Operating System|Server Version|CDI spec|Runtimes|Default Runtime' || true

echo
echo "=== GPU container test ==="
if timeout 90 docker run --rm --gpus all ubuntu:22.04 nvidia-smi; then
  echo
  echo "RESULT: Docker can access the GPU."
  exit 0
fi

echo
echo "RESULT: Docker cannot access the GPU."
echo
echo "Toolkit on the host may still be fine; the usual cause is Docker Desktop"
echo "for Linux, which runs a LinuxKit VM without host GPU passthrough."
echo "Use a real Docker Engine on the host (docker-ce) or Podman with CDI."
echo "See sim/docker/README.md, section 'GPU mode / Docker Desktop'."
exit 1
