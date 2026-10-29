# BloomBee local simulation — full setup

This is the exact environment and package set used to validate the `sim/`
harness on a single machine with one NVIDIA GPU. Follow it if you need to
reproduce the environment from scratch.

> The container setup in `sim/docker/` is self-contained: the image installs
> everything below at build time, so end users only need Docker.

## Tested host

| Component | Value |
|---|---|
| OS | Ubuntu 25.04 |
| GPU | NVIDIA GeForce RTX 3070 (8 GB, CC 8.6) |
| Driver | `580.95.05` (CUDA 13.0 capable) |
| CUDA runtime used by torch | 12.8 (bundled in the PyTorch wheel) |
| Python | 3.11.18 (managed by `uv`) |
| RAM | 32 GB |
| Disk | ~10 GB for the venv + ~2 GB for the HF cache |

`nvcc` / a system CUDA toolkit is **not** required: the CUDA 12.8 runtime
libraries come inside the `torch==2.11.0+cu128` wheel. The NVIDIA **driver**
must be installed on the host (`nvidia-smi` must work).

Python 3.11/3.12 is recommended. Python 3.13 is problematic because the project
pins `numpy<2`, and NumPy 1.26 does not provide Python 3.13 wheels.

## 1. System tools

```bash
sudo apt-get update
sudo apt-get install -y build-essential git wget curl python3.11 python3.11-venv iproute2
```

For a machine without Python 3.11, `uv` can provide it:

```bash
python3 -m venv /tmp/bb-bootstrap
/tmp/bb-bootstrap/bin/pip install uv
UV_PYTHON_INSTALL_DIR="$PWD/sim/.uv-python" \
UV_CACHE_DIR="$PWD/sim/.uv-cache" \
  /tmp/bb-bootstrap/bin/uv python install 3.11
UV_PYTHON_INSTALL_DIR="$PWD/sim/.uv-python" \
UV_CACHE_DIR="$PWD/sim/.uv-cache" \
  /tmp/bb-bootstrap/bin/uv venv --python 3.11 sim/.venv
```

The rest of this document assumes Python is `sim/.venv/bin/python`.

## 2. PyTorch (CUDA)

```bash
sim/.venv/bin/python -m pip install --upgrade pip wheel setuptools "grpcio-tools==1.67.1"

# CUDA 12.8 build (RTX 3070 / driver >= 525)
sim/.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0+cu128"
```

CPU-only alternative:

```bash
sim/.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.15.0+cpu"
```

Verify:

```bash
sim/.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

## 3. Hivemind (pinned commit)

The project pins Hivemind to commit `4bd43b77895019b20d18d81d0d0c1a5ab9a10847`.

```bash
git clone https://github.com/learning-at-home/hivemind.git sim/.deps/hivemind
git -C sim/.deps/hivemind checkout 4bd43b77895019b20d18d81d0d0c1a5ab9a10847

# The pinned tree uses the old setuptools license field.
python - <<'PY'
from pathlib import Path
p = Path("sim/.deps/hivemind/pyproject.toml")
s = p.read_text().replace('license = "MIT"', 'license = {text = "MIT"}')
p.write_text(s)
PY

sim/.venv/bin/python -m pip install \
  --no-build-isolation --no-deps sim/.deps/hivemind
```

## 4. BloomBee itself

```bash
sim/.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

`--no-deps` is intentional: the next step installs the exact dependency set
without letting the old `hivemind @ git+...` requirement re-resolve torch.

## 5. Runtime dependencies

```bash
sim/.venv/bin/python -m pip install \
  "transformers>=5.5.0,<6" "accelerate>=0.27.2" \
  "huggingface-hub>=1.5.0,<2.0.0" "tokenizers>=0.13.3" \
  "safetensors>=0.3.1" sentencepiece pyyaml "numpy==1.26.4" \
  "scipy>=1.2.1" "pydantic>=2.0.0" "protobuf>=5.29.0" \
  grpcio uvloop msgpack sortedcontainers "prefetch_generator>=1.0.1" \
  "configargparse>=1.2.3" "py-multihash>=0.2.3" "cryptography>=3.4.6" \
  "varint>=1.0.2" "base58>=1.0.2" "netaddr>=1.3.0" "idna>=3.10" \
  "py-cid>=0.3.0" "bitsandbytes==0.46.0" "speedtest-cli==2.1.3" \
  "tensor_parallel==1.0.23" humanfriendly "async-timeout>=4.0.2" \
  cpufeature "Dijkstar>=2.6.0" attrs nvidia-ml-py "zstandard>=0.22.0" requests

# peft is imported by the server while creating a backend.  Install it without
# dependencies so pip does not downgrade transformers/torch.
sim/.venv/bin/python -m pip install --no-deps "peft==0.8.2"

# Test tooling
sim/.venv/bin/python -m pip install \
  "pytest>=8.2,<10" pytest-forked "pytest-asyncio>=0.23,<2" \
  pytest-timeout "anyio<4" psutil black==22.3.0 isort==5.10.1
```

### Versions used in the validated run

| Package | Version |
|---|---|
| Python | 3.11.18 |
| torch | 2.11.0+cu128 |
| transformers | 5.21.0 |
| hivemind | 1.2.0.dev0 (`4bd43b7`) |
| accelerate | 1.15.0 |
| huggingface-hub | 1.33.0 |
| tokenizers | 0.23.3 |
| numpy | 1.26.4 |
| pydantic | 2.14.1 |
| protobuf | 5.29.6 |
| grpcio | 1.85.0 |
| peft | 0.8.2 |
| bitsandbytes | 0.46.0 |
| tensor_parallel | 1.0.23 |
| pytest | 9.2.0 |

## 6. Verify and run

```bash
export PYTHON="$PWD/sim/.venv/bin/python"

"$PYTHON" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./sim/simctl doctor
./sim/run.sh
```

The default is `Qwen/Qwen3-0.6B`, two workers (14 layers each), bootstrap DHT on
`127.0.0.1:31340`, workers on `31341`/`31342`, client on CPU.

After startup:

```bash
./sim/simctl status
./sim/simctl ask "Say hello in one sentence." --max-new-tokens 32
./sim/simctl down
```

### CPU-only mode

```bash
./sim/simctl up --device cpu --start-timeout 1800 --smoke-timeout 1200 --max-new-tokens 4
```

The harness sets `BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER=0` for CPU mode and passes
explicit `--block_indices`, so BloomBee never auto-sizes blocks from the full
VRAM. See `sim/README.md` for the CPU validation details.

### Useful overrides

```bash
# larger KV cache for long answers
./sim/run.sh --force --kv-tokens 2048 --max-batch-size 2048

# tighter host memory
BLOOMBEE_COPY_BUFFER_MB=16 ./sim/run.sh

# a different model (must be supported by BloomBee)
./sim/run.sh --model Qwen/Qwen3-1.7B --dtype bfloat16
```

## 7. Troubleshooting

- **`python: command not found`** — use `./sim/simctl` or `$PYTHON`; the wrapper
  automatically prefers `sim/.venv/bin/python`.
- **`torch.cuda.is_available() == False`** — check `nvidia-smi`, the driver, and
  that you installed a `+cuXXX` torch wheel, not `+cpu`.
- **`Cannot allocate KV cache ... max = 512`** — restart with
  `--kv-tokens 2048` (or lower `--max-new-tokens`). `./sim/simctl ask` now
  fails fast instead of retrying forever.
- **High host RAM / `Shmem` in gigabytes** — the fixed FlexGen copy buffer is
  controlled by `BLOOMBEE_COPY_BUFFER_MB` (default 64; was hard-coded to 1 GiB
  per copy worker). Use `./sim/simctl down --all` to kill orphaned workers.
- **First run is slow** — it downloads the model into `sim/.cache/hf`.

## 8. Container mode

See `sim/docker/README.md`. The image is self-contained (Python, torch,
transformers, pinned Hivemind, BloomBee and all runtime dependencies); no host
Python environment is needed. Network isolation/latency is provided by a Docker
bridge plus `tc netem` inside each worker container.
