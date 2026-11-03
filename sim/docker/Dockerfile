# Self-contained BloomBee simulation image.
#
# CPU (default, works on Docker Desktop):
#   docker build -f sim/docker/Dockerfile -t bloombee-sim:cpu .
#
# CUDA 12.8 (host needs the NVIDIA driver + NVIDIA Container Toolkit):
#   docker build -f sim/docker/Dockerfile \
#     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
#     --build-arg TORCH_SPEC=torch==2.11.0+cu128 \
#     -t bloombee-sim:cuda .
#
# Another model (baked into the image, so a rebuild is required):
#   docker build -f sim/docker/Dockerfile \
#     --build-arg MODEL=meta-llama/Llama-3.2-1B \
#     -t bloombee-sim:cpu .
FROM python:3.11-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC=torch==2.11.0+cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/opt/hf \
    HUGGINGFACE_HUB_CACHE=/opt/hf/hub \
    BLOOMBEE_CACHE=/opt/bloombee-cache \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates iproute2 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/bloombee

# Build tooling must be present before hivemind's PEP 517 build.
RUN python -m pip install --upgrade pip wheel setuptools "grpcio-tools==1.67.1"

RUN python -m pip install --index-url "${TORCH_INDEX}" "${TORCH_SPEC}"

COPY sim/docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# Pinned Hivemind commit used by the project.
RUN git clone --filter=blob:none https://github.com/learning-at-home/hivemind.git /tmp/hivemind \
    && git -C /tmp/hivemind checkout 4bd43b77895019b20d18d81d0d0c1a5ab9a10847 \
    && python - <<'PY'
from pathlib import Path

p = Path("/tmp/hivemind/pyproject.toml")
text = p.read_text()
old, new = 'license = "MIT"', 'license = {text = "MIT"}'
if old not in text:
    raise SystemExit("Expected Hivemind pyproject license field not found")
p.write_text(text.replace(old, new, 1))
PY
RUN python -m pip install --no-build-isolation --no-deps /tmp/hivemind \
    && rm -rf /tmp/hivemind

# Bake the default model before copying the project, so code changes do not
# invalidate the (large) model layer. MODEL is declared here so that changing
# it only rebuilds this layer and the ones below, not torch/hivemind.
ARG MODEL=Qwen/Qwen3-0.6B
ENV BLOOMBEE_SIM_MODEL=${MODEL}

RUN python - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(os.environ["BLOOMBEE_SIM_MODEL"])
PY

LABEL org.bloombee.sim.model="${BLOOMBEE_SIM_MODEL}"

# Runtime is fully offline for model/tokenizer access.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Project itself.
COPY . /opt/bloombee
RUN python -m pip install --no-build-isolation --no-deps /opt/bloombee \
    && python -m pip install --no-deps "peft==0.8.2"

COPY sim/docker/entrypoint.sh /usr/local/bin/bloombee-entrypoint
RUN chmod +x /usr/local/bin/bloombee-entrypoint

# Fail the build early if the runtime cannot import the stack.
RUN python - <<'PY'
import torch, transformers, hivemind, bloombee, peft
print("image stack:", torch.__version__, transformers.__version__, bloombee.__version__)
PY

WORKDIR /opt/bloombee
ENTRYPOINT ["/usr/local/bin/bloombee-entrypoint"]
CMD ["server"]
