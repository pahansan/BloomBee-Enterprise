# Containerized BloomBee simulation

A self-contained Docker setup: the image contains Python, PyTorch,
Transformers, pinned Hivemind, BloomBee and the default `Qwen/Qwen3-0.6B`
weights. Runtime containers need **no internet and no host Python packages**.

The swarm runs on an isolated Docker bridge (`172.31.50.0/24`, `internal: true`):

```text
        client .13
            |
   +--------+---------+
   |                  |
node-0 .11        node-1 .12
layers 0:14       layers 14:28
   \                  /
    +---- bootstrap .10 (DHT)
```

Each worker runs `tc netem` on `eth0` at startup, so every link has configurable
delay/jitter/loss/rate. The bootstrap container has no netem.

## Requirements

- Docker Engine or Docker Desktop with Compose v2.
- CPU mode (default): any Docker installation.
- GPU mode: **real Docker Engine on the host** with the NVIDIA driver and
  NVIDIA Container Toolkit (`nvidia-ctk` / `nvidia-container-runtime`).
  Docker Desktop for Linux does **not** expose the host GPU through CDI; use
  CPU mode there, or run the host-native `./sim/run.sh`.

The image is large on purpose (torch + CUDA/CPU libs + model weights). It is
built once and reused.

## Quick start (CPU, no host dependencies)

```bash
./sim/docker/run.sh up
```

This builds `bloombee-sim:cpu` if needed, starts bootstrap + two workers on the
isolated network, waits for all 28 blocks, and runs the smoke generation.

Query the swarm:

```bash
./sim/docker/run.sh ask "Say hello in one sentence." --max-new-tokens 32
./sim/docker/run.sh ask "Напиши короткий рецепт блинов." --max-new-tokens 200 --temperature 0.7
```

Manage:

```bash
./sim/docker/run.sh status
./sim/docker/run.sh logs node-0
./sim/docker/run.sh smoke
./sim/docker/run.sh down
./sim/docker/run.sh down --clean     # also removes state and .env
```

`up` accepts:

```text
--model <hf-id>       --layers <N>       --dht-prefix <prefix>
--kv-tokens <N>       --max-batch <N>
--delay-ms <N>        --jitter-ms <N>    --loss-pct <N>    --rate-mbit <N>
--build               --gpu
```

Example with a harsher WAN-like link:

```bash
./sim/docker/run.sh up --delay-ms 80 --jitter-ms 20 --loss-pct 1 --rate-mbit 20
```

## GPU mode

Prerequisites: NVIDIA driver on the host and the NVIDIA Container Toolkit
installed in Docker (`docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi` must work).

```bash
./sim/docker/run.sh up --gpu
```

This builds `bloombee-sim:cuda` (`torch==2.11.0+cu128`) and starts the workers
with `--device cuda:0 --dtype bfloat16`. The model is still in the image, so
runtime network access is not required.

## Network emulation while running

The compose workers have `NET_ADMIN`; change the qdisc live:

```bash
# add 100 ms delay to node-0's egress
./sim/docker/run.sh netem node-0 delay 100ms

# 2% packet loss
./sim/docker/run.sh netem node-0 loss 2%

# rate limit to 10 Mbit
./sim/docker/run.sh netem node-0 rate 10mbit

# remove shaping
./sim/docker/run.sh netem node-0 reset
```

Because all traffic is on the Docker bridge, this affects only the simulated
swarm, not the host network.

## Isolation properties

- **Filesystem/process isolation**: each worker is its own container with its own
  `/state/node-N` home, TMPDIR and identity.
- **Network isolation**: dedicated internal bridge, no external egress, no host
  port publishing; all traffic is TCP over the bridge.
- **Dependency isolation**: Python/torch/transformers/hivemind/BloomBee and the
  model are inside the image. The host needs only Docker.
- **Resource limits**: `cpus`, `mem_limit`, `shm_size` per service.

Not isolated: the GPU is shared when using `--gpu` (no MIG), and containers on
the same host still share CPU/RAM/GPU bandwidth.

## How the bootstrap address is resolved

The bootstrap PeerID is generated at container start (deterministic only if the
`state` volume is kept). `sim/docker/run.sh up` therefore starts the bootstrap
first, parses its `/ip4/172.31.50.10/tcp/31340/p2p/...` address from the
container logs, writes it to `sim/docker/.env`, and only then starts the
workers/client. `BOOTSTRAP_ADDR` is passed to every service.

## Files

```text
sim/docker/
├── Dockerfile          # self-contained CPU/CUDA image
├── requirements.txt    # runtime packages (torch/hivemind installed separately)
├── entrypoint.sh       # bootstrap / server / client roles + netem
├── compose.yaml        # isolated bridge, static IPs, CPU defaults
├── compose.gpu.yaml    # `gpus: all` override
├── build.sh            # docker build for cpu|cuda
├── run.sh              # up/down/smoke/ask/logs/status/netem
└── README.md
```
