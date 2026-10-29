# Containerized BloomBee simulation

A self-contained Docker swarm: the image contains Python, PyTorch, Transformers,
pinned Hivemind, BloomBee and the default `Qwen/Qwen3-0.6B` weights. Runtime
containers need **no internet and no host Python packages**.

The swarm runs on an isolated Docker bridge (`172.31.50.0/24`, `internal: true`):

```text
        client
            |
   +--------+---------+
   |                  |
node-0            node-1 ...
layers 0:10       10:19 ...
   \                  /
    +---- bootstrap (DHT)
```

Every worker gets a contiguous, equally sized block range and a per-worker
`tc netem` qdisc.

## Requirements

- Docker Engine with Compose v2 on the host.
- The user must be in the `docker` group (`sudo usermod -aG docker "$USER"`,
  then re-login or `newgrp docker`).
- For GPU mode: NVIDIA driver and NVIDIA Container Toolkit.

Docker Desktop for Linux runs a VM without host GPU passthrough; GPU mode needs
host Docker Engine.

## Quick start

```bash
# CPU, 2 workers
./sim/docker/run.sh up

# GPU, 3 workers (layer split is automatic)
./sim/docker/run.sh up --gpu --nodes 3
```

`up` builds the image if it is missing, renders `compose.generated.yaml`, starts
the bootstrap, waits until every block is online, and runs a smoke request.

Ask the model:

```bash
./sim/docker/run.sh ask "Say hello in one sentence." --max-new-tokens 32
./sim/docker/run.sh ask "Write a haiku." --temperature 0.7 --max-new-tokens 64
./sim/docker/run.sh ask "The capital of France is" --raw --max-new-tokens 8
```

Manage:

```bash
./sim/docker/run.sh status
./sim/docker/run.sh logs node-0
./sim/docker/run.sh smoke
./sim/docker/run.sh down
./sim/docker/run.sh down --clean   # also removes state, .env, generated compose
```

## Node count and layer split

```bash
./sim/docker/run.sh up --gpu --nodes 2   # 28 -> 14 + 14
./sim/docker/run.sh up --gpu --nodes 3   # 28 -> 10 + 9 + 9
./sim/docker/run.sh up --gpu --nodes 4   # 28 -> 7 + 7 + 7 + 7
```

`--nodes N` requires `N <= --layers` (default `28`, matching
`Qwen/Qwen3-0.6B`). The remainder goes to the first workers. For another model:

```bash
./sim/docker/run.sh up --gpu --nodes 4 --layers 32 --model <hf-id>
```

`run.sh` writes `compose.generated.yaml` with one service per worker; there is
no static compose file to edit. The `/24` bridge supports roughly 240 workers.

## Build

```bash
./sim/docker/build.sh cpu     # default; works without a GPU
./sim/docker/build.sh cuda    # requires NVIDIA Container Toolkit at runtime
./sim/docker/run.sh up --gpu --build
```

The first CUDA build takes about 15 minutes (torch CUDA + model). Subsequent
builds reuse layers.

## Network emulation

Workers run with `NET_ADMIN`; change the qdisc live:

```bash
./sim/docker/run.sh netem node-0 delay 100ms
./sim/docker/run.sh netem node-0 loss 2%
./sim/docker/run.sh netem node-0 rate 10mbit
./sim/docker/run.sh netem node-0 reset
```

`down --clean` deletes root-owned state through a throw-away root container, so
it does not need `sudo`.

## Benchmarks

```bash
./sim/docker/run.sh bench --seq_len 32
./sim/docker/run.sh bench --seq_len 64 --batch_size 4
```

`bench` runs `benchmarks/benchmark_inference.py` inside a one-off client
container and prints throughput/latency. The repo also has
`benchmarks/benchmark_forward.py`,
`benchmarks/benchmark_speculative_decoding.py` and `benchmarks/compression/`.

Reference: RTX 3070, 3 GPU workers, bf16 servers, CPU client, netem
20ms/0.1%/100Mbit, `seq_len=32` — about `4.2 tokens/sec`, mean latency `~239 ms`.
Server-side per-stage numbers are in `./sim/docker/run.sh logs node-0`
(`[PAPER_TIMING_TABLE]`, `[PIPELINE_GPU2GPU]`).

## Isolation

- Each worker is a separate container with its own `/state/node-N` home and
  TMPDIR.
- Dedicated internal bridge, no external egress and no published ports.
- Python, torch, Hivemind, BloomBee and the model are baked into the image.
- CPU/RAM/shm limits are set per service; GPU is shared when using `--gpu`.

## Bootstrap address

`run.sh up` starts the bootstrap first, parses its
`/ip4/172.31.50.10/tcp/31340/p2p/...` address from the logs, writes
`BOOTSTRAP_ADDR` to `sim/docker/.env`, and only then starts the workers.
`down --clean` removes the identity and generates a new peer ID next time.

## Files

```text
sim/docker/
├── Dockerfile          # self-contained CPU/CUDA image
├── requirements.txt    # runtime packages (torch/hivemind installed separately)
├── entrypoint.sh       # bootstrap / server / client roles + netem
├── render_compose.py   # render N-worker compose.generated.yaml
├── build.sh            # docker build for cpu|cuda
├── run.sh              # up/down/smoke/ask/bench/logs/status/netem
└── README.md
```
