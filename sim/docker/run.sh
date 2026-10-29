#!/usr/bin/env bash
# Containerized BloomBee swarm with an isolated bridge network and netem.
#
#   ./sim/docker/run.sh up                 # CPU image, 2 workers, 20ms link
#   ./sim/docker/run.sh up --nodes 4       # auto-split 28 layers over 4 workers
#   ./sim/docker/run.sh up --gpu --nodes 3 # CUDA image (NVIDIA runtime required)
#   ./sim/docker/run.sh smoke
#   ./sim/docker/run.sh ask "prompt" --max-new-tokens 128
#   ./sim/docker/run.sh bench --seq_len 32
#   ./sim/docker/run.sh logs node-0
#   ./sim/docker/run.sh status
#   ./sim/docker/run.sh down [--clean]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_DIR="$ROOT/sim/docker"
ENV_FILE="$DOCKER_DIR/.env"
GENERATED="$DOCKER_DIR/compose.generated.yaml"
mkdir -p "$DOCKER_DIR/state"
touch "$ENV_FILE"

COMMAND="${1:-up}"
shift || true

MODE="cpu"
USE_GPU=0
BUILD=0
MODEL="Qwen/Qwen3-0.6B"
DHT_PREFIX="sim-Qwen3-0-6B"
KV_TOKENS=2048
MAX_BATCH=2048
NODES=2
LAYERS=28
DTYPE="float32"
DEVICE="cpu"
NETEM_DELAY_MS=20
NETEM_JITTER_MS=5
NETEM_LOSS_PCT=0.1
NETEM_RATE_MBIT=100

load_env() {
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
}
load_env

compose() {
  if [[ ! -f "$GENERATED" ]]; then
    echo "ERROR: no $GENERATED; run '$0 up' first" >&2
    return 2
  fi
  docker compose -f "$GENERATED" --project-directory "$DOCKER_DIR" --env-file "$ENV_FILE" "$@"
}

delete_state_dir() {
  local dir="$DOCKER_DIR/state"
  [[ -e "$dir" ]] || return 0
  if rm -rf "$dir" 2>/dev/null; then
    return 0
  fi
  # Containers run as root, so files under ./state may be root-owned. Delete
  # them through a throw-away root container instead of requiring sudo.
  local image="${IMAGE_TAG:-ubuntu:22.04}"
  docker run --rm -v "$dir:/state" --entrypoint /bin/sh "$image" \
    -c 'find /state -mindepth 1 -delete' >/dev/null 2>&1 || true
  rm -rf "$dir" 2>/dev/null || true
}

write_env() {
  local addr="${1:-}"
  cat > "$ENV_FILE" <<EOF
IMAGE_TAG=${IMAGE_TAG}
DEVICE=${DEVICE}
DTYPE=${DTYPE}
MODEL=${MODEL}
DHT_PREFIX=${DHT_PREFIX}
KV_TOKENS=${KV_TOKENS}
MAX_BATCH=${MAX_BATCH}
NODES=${NODES}
LAYERS=${LAYERS}
BOOTSTRAP_ADDR=${addr}
CLIENT_DEVICE=${CLIENT_DEVICE:-cpu}
CLIENT_DTYPE=${CLIENT_DTYPE:-float32}
NETEM_ENABLE=${NETEM_ENABLE:-1}
NETEM_DELAY_MS=${NETEM_DELAY_MS}
NETEM_JITTER_MS=${NETEM_JITTER_MS}
NETEM_LOSS_PCT=${NETEM_LOSS_PCT}
NETEM_RATE_MBIT=${NETEM_RATE_MBIT}
BLOOMBEE_COPY_BUFFER_MB=${BLOOMBEE_COPY_BUFFER_MB:-64}
EOF
}

wait_for_bootstrap() {
  local addr=""
  for _ in $(seq 1 180); do
    addr="$(compose logs --no-color bootstrap 2>/dev/null \
      | grep -oE '/ip4/172\.31\.50\.10/tcp/31340/p2p/[A-Za-z0-9]+' \
      | tail -1 || true)"
    if [[ -n "$addr" ]]; then
      printf '%s\n' "$addr"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: bootstrap did not announce its peer address; logs:" >&2
  compose logs --no-color --tail=80 bootstrap >&2 || true
  return 1
}

case "$COMMAND" in
  build)
    MODE="${1:-cpu}"
    "$DOCKER_DIR/build.sh" "$MODE"
    ;;

  up)
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --gpu) USE_GPU=1; MODE="cuda"; shift ;;
        --build) BUILD=1; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --dht-prefix) DHT_PREFIX="$2"; shift 2 ;;
        --nodes) NODES="$2"; shift 2 ;;
        --kv-tokens) KV_TOKENS="$2"; shift 2 ;;
        --max-batch) MAX_BATCH="$2"; shift 2 ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --delay-ms) NETEM_DELAY_MS="$2"; shift 2 ;;
        --jitter-ms) NETEM_JITTER_MS="$2"; shift 2 ;;
        --loss-pct) NETEM_LOSS_PCT="$2"; shift 2 ;;
        --rate-mbit) NETEM_RATE_MBIT="$2"; shift 2 ;;
       *) echo "unknown up argument: $1" >&2; exit 2 ;;
      esac
    done

    if [[ "$USE_GPU" == "1" ]]; then
      # Do not let an old .env from a CPU run override the GPU image.
      IMAGE_TAG="bloombee-sim:cuda"
      DEVICE="cuda:0"
      DTYPE="bfloat16"
      CLIENT_DEVICE="cpu"
    else
      IMAGE_TAG="bloombee-sim:cpu"
      DEVICE="cpu"
      DTYPE="float32"
      CLIENT_DEVICE="cpu"
    fi
    export IMAGE_TAG DEVICE DTYPE CLIENT_DEVICE

    if [[ "$BUILD" == "1" ]] || ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
      "$DOCKER_DIR/build.sh" "$MODE"
    fi

    if ! [[ "$NODES" =~ ^[0-9]+$ ]] || (( NODES < 1 )); then
      echo "ERROR: --nodes must be a positive integer" >&2
      exit 2
    fi
    if (( NODES > LAYERS )); then
      echo "ERROR: cannot split $LAYERS layers between $NODES nodes" >&2
      exit 2
    fi

    if [[ -f "$GENERATED" ]]; then
      compose down --remove-orphans >/dev/null 2>&1 || true
    fi
    write_env ""
    gpu_arg=()
    if [[ "$USE_GPU" == "1" ]]; then gpu_arg=(--gpu); fi
    python3 "$DOCKER_DIR/render_compose.py" \
      --nodes "$NODES" --layers "$LAYERS" \
      --image-tag "$IMAGE_TAG" \
      --model "$MODEL" --dht-prefix "$DHT_PREFIX" \
      --device "$DEVICE" --dtype "$DTYPE" \
      --client-device "$CLIENT_DEVICE" --client-dtype "${CLIENT_DTYPE:-float32}" \
      --kv-tokens "$KV_TOKENS" --max-batch "$MAX_BATCH" \
      --copy-buffer-mb "${BLOOMBEE_COPY_BUFFER_MB:-64}" \
      --delay-ms "$NETEM_DELAY_MS" --jitter-ms "$NETEM_JITTER_MS" \
      --loss-pct "$NETEM_LOSS_PCT" --rate-mbit "$NETEM_RATE_MBIT" \
      "${gpu_arg[@]}" \
      --output "$GENERATED"
    compose up -d --remove-orphans bootstrap
    addr="$(wait_for_bootstrap)"
    write_env "$addr"
    echo "[up] bootstrap: $addr"
    node_services=()
    for ((i=0; i<NODES; i++)); do node_services+=("node-$i"); done
    compose up -d --remove-orphans "${node_services[@]}"
    echo "[up] $NODES worker(s) started; running readiness + smoke"
    compose run --rm --no-deps client
    echo
    echo "Container swarm is UP"
    echo "  nodes    : $NODES"
    echo "  bootstrap: $addr"
    compose ps
    echo
    echo "Try:  $0 ask \"Say hello in one sentence.\" --max-new-tokens 32"
    echo "Bench: $0 bench --seq_len 32"
    echo "Stop: $0 down"
    ;;

  down)
    CLEAN=0
    if [[ "${1:-}" == "--clean" ]]; then CLEAN=1; fi
    load_env
    if [[ -f "$GENERATED" ]]; then
      compose down --remove-orphans
    fi
    if [[ "$CLEAN" == "1" ]]; then
      delete_state_dir
      rm -f "$ENV_FILE" "$GENERATED"
      echo "[down] removed container state, .env and generated compose (image kept)"
    else
      echo "[down] state and .env kept"
    fi
    ;;

  smoke)
    load_env
    [[ -n "${BOOTSTRAP_ADDR:-}" ]] || { echo "ERROR: run '$0 up' first" >&2; exit 2; }
    compose run --rm --no-deps client
    ;;
  bench)
    load_env
    [[ -n "${BOOTSTRAP_ADDR:-}" ]] || { echo "ERROR: run '$0 up' first" >&2; exit 2; }
    bench_args=("$@")
    if [[ ! " ${bench_args[*]} " =~ --seq_len ]]; then
      bench_args+=(--seq_len "${BENCH_SEQ_LEN:-32}" --early_terminate_steps "${BENCH_SEQ_LEN:-32}")
    fi
    compose run --rm --no-deps client python benchmarks/benchmark_inference.py \
      --model "$MODEL" \
      --initial_peers "$BOOTSTRAP_ADDR" \
      --dht_prefix "$DHT_PREFIX" \
      --torch_dtype "${CLIENT_DTYPE:-float32}" \
      --client_device "${CLIENT_DEVICE:-cpu}" \
      "${bench_args[@]}"
    ;;


  ask)
    load_env
    [[ -n "${BOOTSTRAP_ADDR:-}" ]] || { echo "ERROR: run '$0 up' first" >&2; exit 2; }
    PROMPT="${1:?usage: $0 ask \"prompt\" [--max-new-tokens N]}"
    shift
    MAX_NEW=64
    TEMP=0.0
    TOP_P=0.9
    RAW=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --max-new-tokens) MAX_NEW="$2"; shift 2 ;;
        --temperature) TEMP="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --raw) RAW=1; shift ;;
        *) echo "unknown ask argument: $1" >&2; exit 2 ;;
      esac
    done
    cmd=(python sim/ask_client.py
      --model "$MODEL"
      --initial-peers "$BOOTSTRAP_ADDR"
      --dht-prefix "$DHT_PREFIX"
      --device "${CLIENT_DEVICE:-cpu}"
      --dtype "${CLIENT_DTYPE:-float32}"
      --prompt "$PROMPT"
      --max-new-tokens "$MAX_NEW"
      --max-cache-tokens "$KV_TOKENS"
      --temperature "$TEMP"
      --top-p "$TOP_P")
    [[ "$RAW" == "1" ]] && cmd+=(--raw)
    compose run --rm --no-deps client "${cmd[@]}"
    ;;

  logs)
    load_env
    NAME="${1:-}"
    if [[ -n "$NAME" ]]; then
      compose logs --no-color --tail=200 "$NAME"
    else
      compose logs --no-color --tail=120
    fi
    ;;

  status)
    load_env
    compose ps
    echo
    mapfile -t ids < <(compose ps -q 2>/dev/null || true)
    if [[ ${#ids[@]} -gt 0 && -n "${ids[0]}" ]]; then
      docker stats --no-stream --format \
        'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}' \
        "${ids[@]}"
    fi
    ;;

  netem)
    load_env
    NODE="${1:?usage: $0 netem node-0|node-1 delay|loss|rate|reset value}"
    ACTION="${2:?}"
    shift 2
    case "$ACTION" in
      delay) compose exec "$NODE" tc qdisc change dev eth0 root netem delay "$1" ;;
      loss) compose exec "$NODE" tc qdisc change dev eth0 root netem loss "$1" ;;
      rate) compose exec "$NODE" tc qdisc change dev eth0 root netem rate "$1" ;;
      reset) compose exec "$NODE" tc qdisc del dev eth0 root 2>/dev/null || true ;;
      *) echo "unknown netem action: $ACTION" >&2; exit 2 ;;
    esac
    ;;

  exec)
    load_env
    compose exec "$@"
    ;;

  *)
    echo "usage: $0 {build|up|down|smoke|ask|bench|logs|status|netem|exec}" >&2
    exit 2
    ;;
esac
