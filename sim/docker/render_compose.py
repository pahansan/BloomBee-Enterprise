#!/usr/bin/env python3
"""Render a Docker Compose file for N BloomBee workers.

The host-side ``sim/simctl.py`` already splits layers evenly for N processes.
This script does the same for containers: it allocates contiguous, equal-sized
block ranges, gives every worker a static bridge IP and writes a complete
compose file. Standard library only, so it also works with the system python3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUBNET_PREFIX = "172.31.50."
BOOTSTRAP_IP = SUBNET_PREFIX + "10"


def split_layers(layers: int, nodes: int) -> list[tuple[int, int]]:
    if nodes < 1:
        raise ValueError("--nodes must be >= 1")
    if layers < 1:
        raise ValueError("--layers must be >= 1")
    if nodes > layers:
        raise ValueError(f"cannot split {layers} layers between {nodes} nodes")
    base, remainder = divmod(layers, nodes)
    ranges = []
    start = 0
    for index in range(nodes):
        size = base + (1 if index < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def build_compose(args: argparse.Namespace) -> str:
    ranges = split_layers(args.layers, args.nodes)
    client_ip = f"{SUBNET_PREFIX}{11 + args.nodes}"
    if 11 + args.nodes > 254:
        raise ValueError("too many nodes for a /24 bridge network")

    common_env = [
        ("MODEL", args.model),
        ("DHT_PREFIX", args.dht_prefix),
        ("DEVICE", args.device),
        ("DTYPE", args.dtype),
        ("KV_TOKENS", str(args.kv_tokens)),
        ("MAX_BATCH", str(args.max_batch)),
        ("LAYERS", str(args.layers)),
        ("BOOTSTRAP_ADDR", "${BOOTSTRAP_ADDR:-}"),
        ("HF_HOME", "/opt/hf"),
        ("HUGGINGFACE_HUB_CACHE", "/opt/hf/hub"),
        ("BLOOMBEE_CACHE", "/opt/hf/hub"),
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("BLOOMBEE_COPY_BUFFER_MB", args.copy_buffer_mb),
        ("BLOOMBEE_SKIP_VERSION_CHECK", "1"),
        ("NETEM_ENABLE", "1"),
        ("NVIDIA_DRIVER_CAPABILITIES", "compute,utility"),
    ]

    def env_block(extra: list[tuple[str, str]]) -> list[str]:
        result = ["    environment:"]
        for key, value in common_env + extra:
            result.append(f'      {key}: "{value}"')
        return result

    lines = [
        "name: bloombee-sim",
        "",
        "services:",
        "  bootstrap:",
        f"    image: {args.image_tag}",
        '    container_name: bloombee-sim-bootstrap',
        '    restart: "no"',
        '    shm_size: "2gb"',
        "    volumes:",
        "      - ./state:/state",
        "    mem_limit: 1g",
        "    networks:",
        "      simnet:",
        f"        ipv4_address: {BOOTSTRAP_IP}",
    ]
    lines += env_block([("ROLE", "bootstrap"), ("ANNOUNCE_IP", BOOTSTRAP_IP)])

    for index, (start, end) in enumerate(ranges):
        ip = f"{SUBNET_PREFIX}{11 + index}"
        lines += [
            f"  node-{index}:",
            f"    image: {args.image_tag}",
            f"    container_name: bloombee-sim-node-{index}",
            '    restart: "no"',
            '    shm_size: "2gb"',
            '    cap_add: ["NET_ADMIN"]',
            "    volumes:",
            "      - ./state:/state",
            '    cpus: "2"',
            "    mem_limit: 6g",
            "    networks:",
            "      simnet:",
            f"        ipv4_address: {ip}",
        ]
        if args.gpu:
            lines.append("    gpus: all")
        lines += env_block(
            [
                ("ROLE", "server"),
                ("NODE_INDEX", str(index)),
                ("BLOCKS", f"{start}:{end}"),
                ("ANNOUNCE_IP", ip),
                ("NETEM_DELAY_MS", str(args.delay_ms)),
                ("NETEM_JITTER_MS", str(args.jitter_ms)),
                ("NETEM_LOSS_PCT", str(args.loss_pct)),
                ("NETEM_RATE_MBIT", str(args.rate_mbit)),
            ]
        )

    lines += [
        "  client:",
        f"    image: {args.image_tag}",
        "    container_name: bloombee-sim-client",
        '    restart: "no"',
        '    shm_size: "2gb"',
        "    volumes:",
        "      - ./state:/state",
        '    cpus: "2"',
        "    mem_limit: 4g",
        "    networks:",
        "      simnet:",
        f"        ipv4_address: {client_ip}",
    ]
    lines += env_block(
        [
            ("ROLE", "client"),
            ("CLIENT_DEVICE", args.client_device),
            ("CLIENT_DTYPE", args.client_dtype),
            ("CLIENT_NETEM", "0"),
            ("WAIT_TIMEOUT", "600"),
            ("SMOKE_TOKENS", "8"),
            ("SMOKE_PROMPT", "The capital of France is"),
        ]
    )
    lines += [
        "",
        "networks:",
        "  simnet:",
        "    driver: bridge",
        "    internal: true",
        "    ipam:",
        "      config:",
        "        - subnet: 172.31.50.0/24",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dht-prefix", default="sim-Qwen3-0-6B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--client-device", default="cpu")
    parser.add_argument("--client-dtype", default="float32")
    parser.add_argument("--kv-tokens", type=int, default=2048)
    parser.add_argument("--max-batch", type=int, default=2048)
    parser.add_argument("--copy-buffer-mb", default="64")
    parser.add_argument("--delay-ms", default="20")
    parser.add_argument("--jitter-ms", default="5")
    parser.add_argument("--loss-pct", default="0.1")
    parser.add_argument("--rate-mbit", default="100")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        text = build_compose(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({args.nodes} worker(s), {args.layers} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
