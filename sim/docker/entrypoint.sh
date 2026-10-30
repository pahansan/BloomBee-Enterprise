#!/usr/bin/env bash
# Entrypoint for all BloomBee simulation containers.
#
# ROLE=bootstrap | server | client
set -euo pipefail

ROLE="${ROLE:-${1:-server}}"
if [[ $# -gt 0 && ( "$1" == "bootstrap" || "$1" == "server" || "$1" == "client" ) ]]; then
  shift
fi

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DHT_PREFIX="${DHT_PREFIX:-sim-Qwen3-0-6B}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
KV_TOKENS="${KV_TOKENS:-2048}"
MAX_BATCH="${MAX_BATCH:-2048}"
BOOTSTRAP_ADDR="${BOOTSTRAP_ADDR:-}"
ANNOUNCE_IP="${ANNOUNCE_IP:-}"
LAYERS="${LAYERS:-28}"

state_root=/state
mkdir -p "$state_root"

if [[ "$ROLE" == "bootstrap" ]]; then
  mkdir -p "$state_root/bootstrap"
  export HOME="$state_root/bootstrap"
  export TMPDIR="$state_root/bootstrap/tmp"
  mkdir -p "$TMPDIR"
  exec python -m bloombee.cli.run_dht \
    --host_maddrs /ip4/0.0.0.0/tcp/31340 \
    --announce_maddrs "/ip4/${ANNOUNCE_IP}/tcp/31340" \
    --identity_path "$state_root/bootstrap.id" \
    --no_relay \
    --refresh_period 10
fi

if [[ -z "$BOOTSTRAP_ADDR" ]]; then
  echo "ERROR: BOOTSTRAP_ADDR is empty; start the bootstrap container first" >&2
  exit 2
fi

apply_netem() {
  if [[ "${NETEM_ENABLE:-1}" != "1" ]]; then
    return
  fi
  local args=(delay "${NETEM_DELAY_MS:-20}ms" "${NETEM_JITTER_MS:-5}ms" loss "${NETEM_LOSS_PCT:-0.1}%")
  if [[ "${NETEM_RATE_MBIT:-100}" != "0" ]]; then
    args+=(rate "${NETEM_RATE_MBIT}mbit")
  fi
  tc qdisc replace dev eth0 root netem "${args[@]}"
  echo "[netem] eth0: ${args[*]}"
}

if [[ "$ROLE" == "server" ]]; then
  apply_netem
  node_dir="$state_root/node-${NODE_INDEX:?NODE_INDEX is required}"
  mkdir -p "$node_dir/home" "$node_dir/tmp"
  export HOME="$node_dir/home"
  export TMPDIR="$node_dir/tmp"
  export BLOOMBEE_COPY_BUFFER_MB="${BLOOMBEE_COPY_BUFFER_MB:-64}"
  if [[ "$DEVICE" == "cpu" ]]; then
    export BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER=0
    export BLOOMBEE_ENABLE_MICROBATCH_PIPELINE=0
  fi
  exec python -m bloombee.cli.run_server "$MODEL" \
    --initial_peers "$BOOTSTRAP_ADDR" \
    --identity_path "$state_root/node-${NODE_INDEX}.id" \
    --host_maddrs /ip4/0.0.0.0/tcp/31340 \
    --announce_maddrs "/ip4/${ANNOUNCE_IP}/tcp/31340" \
    --block_indices "${BLOCKS:?BLOCKS is required}" \
    --dht_prefix "$DHT_PREFIX" \
    --device "$DEVICE" \
    --torch_dtype "$DTYPE" \
    --num_handlers 1 \
    --prefetch_batches 1 \
    --sender_threads 1 \
    --max_batch_size "$MAX_BATCH" \
    --inference_max_length "$KV_TOKENS" \
    --attn_cache_tokens "$KV_TOKENS" \
    --max_alloc_timeout 60 \
    --w_gpu_percent 100 \
    --cache_gpu_percent 100 \
    --update_period 15 \
    --expiration 60 \
    --request_timeout 120 \
    --session_timeout 600 \
    --step_timeout 120 \
    --throughput 1.0 \
    --no_auto_relay \
    --skip_reachability_check
fi

if [[ "$ROLE" == "client" ]]; then
  mkdir -p "$state_root/client/home" "$state_root/client/tmp"
  export HOME="$state_root/client/home"
  export TMPDIR="$state_root/client/tmp"
  python sim/wait_ready.py \
    --initial-peers "$BOOTSTRAP_ADDR" \
    --dht-prefix "$DHT_PREFIX" \
    --layers "$LAYERS" \
    --timeout "${WAIT_TIMEOUT:-600}"
  if [[ $# -gt 0 ]]; then
    exec "$@"
  fi
  exec python sim/ask_client.py \
    --model "$MODEL" \
    --initial-peers "$BOOTSTRAP_ADDR" \
    --dht-prefix "$DHT_PREFIX" \
    --device "${CLIENT_DEVICE:-cpu}" \
    --dtype "${CLIENT_DTYPE:-$DTYPE}" \
    --max-new-tokens "${SMOKE_TOKENS:-8}" \
    --max-cache-tokens "$KV_TOKENS" \
    --raw \
    --prompt "${SMOKE_PROMPT:-The capital of France is}"
fi

echo "ERROR: unknown ROLE=$ROLE" >&2
exit 2
