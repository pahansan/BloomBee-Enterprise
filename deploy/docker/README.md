# BloomBee image

Code only — weights are a runtime mount (`-v /srv/hf:/opt/hf`). One image serves
bootstrap, worker and client; a node loads only the shards behind its `--block_indices`.

## Build

```bash
./build.sh cpu
./build.sh cu128
DRY_RUN=1 ./build.sh cu128     # print the tag, build nothing
TAG_SHA=1 ./build.sh cu128     # add git sha to tag
```

Tag = `<__version__>-<flavor>`.

## Run

Notes:

- `$ANNOUNCE_IP` — address other peers dial you at: your LAN IP (e.g. `192.168.3.2`) or a public IP with a forwarded port;
- `$BOOTSTRAP` — the `/ip4/<ip>/tcp/31340/p2p/<id>` multiaddr printed by the bootstrap; every worker and client passes it to `--initial_peers`.

Bootstrap (no GPU; can share a host with a worker — use a different port):

```bash
docker run -d --name bootstrap --network host -v /srv/hf:/opt/hf bloombee:0.1.0-cu128 \
  python -m bloombee.cli.run_dht \
    --host_maddrs /ip4/0.0.0.0/tcp/31340 \
    --announce_maddrs /ip4/$ANNOUNCE_IP/tcp/31340 \
    --identity_path /opt/hf/bootstrap.id --no_relay --refresh_period 10
# copy the printed /ip4/<ip>/tcp/31340/p2p/<id> into $BOOTSTRAP
```

Worker:

```bash
docker run --name bb --gpus all --network host --ipc=host \
  -v /srv/hf:/opt/hf bloombee:0.1.0-cu128 \
  python -m bloombee.cli.run_server Qwen/Qwen3-0.6B \
    --initial_peers "$BOOTSTRAP" --block_indices 0:28 \
    --identity_path /opt/hf/node-0.id \
    --host_maddrs /ip4/0.0.0.0/tcp/31341 \
    --announce_maddrs /ip4/$ANNOUNCE_IP/tcp/31341 \
    --cache_dir /opt/hf/bloombee --torch_dtype bfloat16 --throughput eval \
    --batch_size 32  # max batch accepted; client batch must not exceed it
```

## Benchmark

Fixed-workload client that reports per-step latency and throughput.
See `README.logging.md` for what the numbers mean.

CPU:

```bash
docker run --rm --network host -v /srv/hf:/opt/hf bloombee:0.1.0-cu128 \
  python benchmarks/benchmark_inference.py --model Qwen/Qwen3-0.6B \
    --initial_peers "$BOOTSTRAP" --dht_prefix "Qwen/Qwen3-0-6B" \
    --torch_dtype float32 --client_device cpu --prompt_len 32 --seq_len 128 \
    --batch_size 32
```

GPU:

```bash
docker run --rm --gpus all --network host -v /srv/hf:/opt/hf bloombee:0.1.0-cu128 \
  python benchmarks/benchmark_inference.py --model Qwen/Qwen3-0.6B \
    --initial_peers "$BOOTSTRAP" --dht_prefix "Qwen/Qwen3-0-6B" \
    --torch_dtype bfloat16 --client_device cuda --prompt_len 32 --seq_len 128 \
    --batch_size 32
```

## Publish

TODO: Add CI/CD instead of manual building and publishing.

```bash
docker tag bloombee:0.1.0-cu128 ghcr.io/tobusoo/bloombee:0.1.0-cu128
docker push ghcr.io/tobusoo/bloombee:0.1.0-cu128
docker inspect -f '{{index .RepoDigests 0}}' ghcr.io/tobusoo/bloombee:0.1.0-cu128   # @sha256:…

# on a node:
docker pull ghcr.io/tobusoo/bloombee:0.1.0-cu128
```

## GPUs

```bash
docker run --rm --gpus all ghcr.io/tobusoo/bloombee:0.1.0-cu128 python -c \
 "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"
```
