# BloomBee worker/client image - code only, weights are a runtime mount (-v /srv/hf:/opt/hf).
#   deploy/docker/build.sh cpu
#   deploy/docker/build.sh cu128
ARG PYTHON_VERSION=3.11
ARG GIT_SHA=unknown
ARG BLOOMBEE_VERSION=dev
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_SPEC=torch==2.11.0+cu128
# Hivemind: main @ 2026-01-04 "Make Hivemind compatible with PEP 517 (#670)".
# That commit is not in any release tag (1.1.12 -> 1a6ff51), which is why the SHA is pinned.
ARG HIVEMIND_REPO=https://github.com/learning-at-home/hivemind.git
ARG HIVEMIND_REF=4bd43b77895019b20d18d81d0d0c1a5ab9a10847

FROM python:${PYTHON_VERSION}-slim-bookworm AS build
ARG TORCH_INDEX TORCH_SPEC HIVEMIND_REPO HIVEMIND_REF
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/docker/requirements.txt /tmp/req.txt
RUN --mount=type=cache,target=/root/.cache/pip set -eux; \
    pip install -q -U pip wheel setuptools "grpcio-tools==1.67.1"; \
    pip wheel -q -w /w --index-url "$TORCH_INDEX" "$TORCH_SPEC"; \
    printf 'torch==%s\n' "${TORCH_SPEC#torch==}" > /tmp/constraint.txt; \
    pip wheel -q -w /w --find-links /w -c /tmp/constraint.txt -r /tmp/req.txt; \
    git clone -q --filter=blob:none "$HIVEMIND_REPO" /hm; \
    git -C /hm checkout -q "$HIVEMIND_REF"; \
    git -C /hm log -1 --format='hivemind %H %cs %s'; \
    [ -f /hm/pyproject.toml ] && sed -i 's/license = "MIT"/license = {text = "MIT"}/' /hm/pyproject.toml || true; \
    pip wheel -q -w /w --no-build-isolation --no-deps /hm; \
    pip wheel -q -w /w --no-deps "peft==0.8.2"
COPY . /src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel -q -w /w --no-build-isolation --no-deps /src

FROM python:${PYTHON_VERSION}-slim-bookworm
ARG TORCH_SPEC
ARG GIT_SHA
ARG BLOOMBEE_VERSION
LABEL org.opencontainers.image.title="bloombee" \
      org.opencontainers.image.source="https://github.com/pahansan/BloomBee-Enterprise" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${BLOOMBEE_VERSION}" \
      org.bloombee.torch="${TORCH_SPEC}"
ENV PYTHONUNBUFFERED=1 HF_HOME=/opt/hf HUGGINGFACE_HUB_CACHE=/opt/hf/hub \
    BLOOMBEE_CACHE=/opt/hf/bloombee BLOOMBEE_SKIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/docker/requirements.txt /tmp/req.txt
RUN --mount=type=bind,from=build,source=/w,target=/w set -eux; \
    pip install -q --no-index --find-links=/w -r /tmp/req.txt; \
    pip install -q --no-index --find-links=/w --no-deps "$TORCH_SPEC" hivemind bloombee "peft==0.8.2"; \
    rm -rf /tmp/req.txt /root/.cache /usr/local/lib/python3.*/site-packages/torch/include \
           /usr/local/lib/python3.*/site-packages/torch/test /usr/local/lib/python3.*/site-packages/nvidia/*/include; \
    python -c "import torch, transformers, hivemind, bloombee; print(torch.__version__, transformers.__version__)"
WORKDIR /opt/bloombee
# benchmarks/ and sim/ are not installed as a package, but the client role needs them.
COPY --from=build /src/benchmarks /opt/bloombee/benchmarks
COPY --from=build /src/sim /opt/bloombee/sim
CMD ["python", "-m", "bloombee.cli.run_server", "--help"]
