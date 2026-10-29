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
- The user must be in the `docker` group: `sudo usermod -aG docker "$USER"`, then
  re-login (or `newgrp docker`). A shell started before the group change still
  gets `permission denied` on `/var/run/docker.sock`.
- CPU mode (default): any Docker installation.
- GPU mode: **real Docker Engine on the host** with the NVIDIA driver and
  NVIDIA Container Toolkit (`nvidia-ctk` / `nvidia-container-runtime`).
  Docker Desktop for Linux does **not** expose the host GPU through CDI; use
  CPU mode there, or run the host-native `./sim/run.sh`.

The image is large on purpose (torch + CUDA/CPU libs + model weights). It is
built once and reused.

## Quick start (CPU, no host dependencies)

```bash
./sim/docker/run.sh up                 # 2 workers
./sim/docker/run.sh up --nodes 4       # 4 workers, layers split automatically
./sim/docker/run.sh up --gpu --nodes 3 # 3 CUDA workers
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
--nodes <N>           --layers <N>       --model <hf-id>
--dht-prefix <prefix> --kv-tokens <N>    --max-batch <N>
--delay-ms <N>        --jitter-ms <N>    --loss-pct <N>    --rate-mbit <N>
--build               --gpu
```

Example with a harsher WAN-like link:

```bash
./sim/docker/run.sh up --delay-ms 80 --jitter-ms 20 --loss-pct 1 --rate-mbit 20
```

## Node count and layer split

`--nodes N` starts N worker containers and splits the model's blocks evenly:

```bash
# 28 layers -> 14 + 14
./sim/docker/run.sh up --nodes 2

# 28 layers -> 10 + 9 + 9
./sim/docker/run.sh up --nodes 3

# 28 layers -> 7 + 7 + 7 + 7
./sim/docker/run.sh up --gpu --nodes 4
```

The split is contiguous and covers every layer exactly once; the remainder goes
to the first workers (same algorithm as the host-side `sim/simctl.py`). The
number of layers comes from `--layers` (default `28`, matching
`Qwen/Qwen3-0.6B`); pass `--layers` explicitly for another model.

`run.sh` renders `compose.generated.yaml` with one service per worker (static
IPs, block range, netem). `compose.yaml` remains the fixed 2-worker example for
manual Compose use. Limits: `--nodes <= --layers`, and a `/24` bridge allows up
to ~240 workers.

The host-native harness already supports the same option:

```bash
./sim/run.sh --nodes 4
```

## GPU mode

Prerequisites: NVIDIA driver on the host and the NVIDIA Container Toolkit
installed in **the Docker daemon that will run the containers**.

Run the diagnostic first:

```bash
./sim/docker/check_gpu.sh
```

It checks the host GPU, `nvidia-ctk`/`nvidia-container-cli`, the generated CDI
devices, the Docker context, and finally runs a real `--gpus all` container.

### Docker Desktop for Linux cannot pass the GPU

A common and confusing failure:

```text
docker: Error response from daemon: failed to discover GPU vendor from CDI:
no known GPU vendor found
```

This does **not** mean the toolkit is broken. On a typical Docker Desktop
install the host side looks fine:

```text
nvidia-ctk cdi list
# nvidia.com/gpu=0
# nvidia.com/gpu=GPU-...
# nvidia.com/gpu=all

nvidia-container-cli info
# Model: NVIDIA GeForce RTX 3070 ...
```

but the Docker daemon reports:

```text
Operating System: Docker Desktop
Kernel Version: ...-linuxkit
```

Docker Desktop for Linux runs a LinuxKit VM and does not expose the host
`/dev/nvidia*` devices. The toolkit and CDI spec live on the host, not inside
that VM, so `--gpus all` cannot work there.

### Docker Desktop and host Docker Engine can coexist

You do **not** need to uninstall Docker Desktop. On Linux the two daemons use
different sockets and are selected with Docker contexts:

```text
default        -> host Docker Engine   /var/run/docker.sock
desktop-linux  -> Docker Desktop       ~/.docker/desktop/docker.sock
```

Install Docker Engine alongside Desktop, then switch context only for the GPU
container commands:

```bash
docker context ls
docker context use default          # host Engine (GPU capable)
./sim/docker/check_gpu.sh
./sim/docker/build.sh cuda
./sim/docker/run.sh up --gpu
docker context use desktop-linux    # switch back to Desktop for its workloads
```

Alternatively, keep the context unchanged and prefix commands:

```bash
DOCKER_CONTEXT=default ./sim/docker/run.sh up --gpu
```

Important consequences:

- **Images are per-daemon.** The CPU image built in Docker Desktop is not visible
  to the host Engine and vice versa. `run.sh up` rebuilds the image in the
  selected context if it is missing; a CUDA build is required for `--gpu` anyway.
- **Containers/networks are per-daemon.** The isolated `bloombee-sim_simnet`
  network and the running nodes exist only in the daemon that started them.
- **No port conflict**: the simulated swarm publishes no host ports.
- Both daemons can run at the same time; stop Desktop when you want to save RAM
  (`docker desktop stop`, or quit the app). The host Engine keeps working.
- If `docker context use default` says the socket is missing, start the Engine:
  `sudo systemctl enable --now docker`.

### Use host Docker Engine instead

Install Docker Engine on the host (Ubuntu example; add the official Docker apt
repository first if `docker-ce` is not found):

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker
```

Then make sure the CDI spec is visible and verify:

```bash
docker context use default
docker info | grep -E 'Operating System|CDI spec'
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # if needed
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

Once that passes:

```bash
./sim/docker/check_gpu.sh
./sim/docker/build.sh cuda
./sim/docker/run.sh up --gpu
```

`up --gpu` builds `bloombee-sim:cuda` (`torch==2.11.0+cu128`) if needed and
starts workers with `--device cuda:0 --dtype bfloat16`. Verified on host Docker
Engine 29.2.1 + RTX 3070: `torch.cuda.is_available() == True` inside a worker,
GPU memory is used, smoke and `ask` pass. The model is still baked
into the image, so runtime network access is not required.

### Podman alternative

Podman can use CDI directly and does not need a daemon:

```bash
sudo apt-get install -y podman
podman run --rm --device nvidia.com/gpu=all ubuntu:22.04 nvidia-smi
```

The Compose files in this directory target Docker; use host Docker Engine for
the one-command flow.

## Benchmarks

The repository ships benchmark scripts in `benchmarks/`:

| Script | What it measures |
|---|---|
| `benchmark_inference.py` | end-to-end decode throughput and client-observed per-token latency |
| `benchmark_forward.py` | forward-pass latency/throughput |
| `benchmark_speculative_decoding.py` | speculative decoding / batch behavior |
| `benchmarks/compression/*` | lossless codec and compression experiments |

Run inference benchmark against the container swarm:

```bash
./sim/docker/run.sh bench --seq_len 32 --batch_size 1 --warmup_steps 1
```

`bench` waits for the DHT, runs `benchmark_inference.py` inside a one-off
`client` container with the current model, bootstrap address and DHT prefix, and
then removes the client. Extra flags are forwarded to the benchmark:

```bash
./sim/docker/run.sh bench --seq_len 64 --batch_size 4 --n_processes 2
```

Reference run on this host (RTX 3070, 3 GPU workers, bf16 servers, CPU client,
netem 20ms/0.1%/100Mbit, `seq_len=32`):

```text
throughput = 4.19 tokens/sec/sequence
mean latency = 238.64 ms, median = 239.46 ms, p95 = 244.38 ms
```

These numbers include the client-side embeddings/LM head and the simulated
network; they are not a pure GPU kernel benchmark. Server-side per-stage numbers
are in `./sim/docker/run.sh logs node-0` (`[PAPER_TIMING_TABLE]`,
`[PIPELINE_GPU2GPU]`).

For the host-native swarm:

```bash
./sim/simctl env
sim/.venv/bin/python benchmarks/benchmark_inference.py \
  --model Qwen/Qwen3-0.6B --initial_peers "$INITIAL_PEERS" \
  --dht_prefix sim-Qwen3-0-6B --seq_len 32
```

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
├── compose.gpu.yaml    # `gpus: all` override for the fixed compose.yaml
├── render_compose.py   # generate N-node compose.generated.yaml
├── build.sh            # docker build for cpu|cuda
├── run.sh              # up/down/smoke/ask/logs/status/netem
└── README.md
```
