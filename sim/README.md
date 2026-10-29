# Local BloomBee simulation (single machine, single GPU)

A one-click development stand for BloomBee. It starts a private DHT bootstrap
node, several real worker processes with a **manually partitioned** transformer,
waits until every block is registered in the DHT, and runs an end-to-end
generation smoke test. If anything is wrong it exits non-zero and prints the
relevant logs instead of hanging.

This is intended for day-to-day development and for the distributed-systems
course project ("BloomBee enterprise optimisation"), not as a production
deployment.

> Full version-pinned setup (Python, CUDA torch, pinned Hivemind, all runtime packages): **[SETUP.md](SETUP.md)**.

## Requirements

- Linux (tested design) or macOS; Windows via WSL.
- A Python environment with the project installed:

  ```bash
  conda activate bb          # or any venv
  pip install -e .
  ```

- PyTorch with CUDA for real workers. CPU-only mode is supported for
  control-plane tests (`--device cpu`), but generation will be slow.
- One GPU is enough. All workers share it; VRAM is **not** isolated between
  processes (see [Limitations](#limitations)).

The default model is `Qwen/Qwen3-0.6B`: public, small (~1.2 GB), 28 layers, and
explicitly supported by BloomBee. Override with `--model`.

If this checkout already contains the validation environment at `sim/.venv`
(created during setup, gitignored), point the harness at it instead of relying
on an activated conda env:

```bash
export PYTHON="$PWD/sim/.venv/bin/python"
# CUDA check (run this outside the restricted agent sandbox):
"$PYTHON" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## Quick start

```bash
./sim/run.sh
```

Equivalent to:

```bash
./sim/simctl up
```

> Use `./sim/simctl ...` rather than `python3 sim/simctl.py ...`. The wrapper
> picks `sim/.venv` (or `$PYTHON`) automatically. `status`, `logs` and `down`
> are stdlib-only and happen to work under a system `python3`, but `up`/`smoke`
> spawn workers with that interpreter and will fail without torch/BloomBee.

Expected output (abridged):

```
Environment
  python       : 3.11.x
  torch        : 2.x.x
  transformers : 5.x.x
  hivemind     : ...
  bloombee     : 2.3.0.dev2
Plan
  model        : Qwen/Qwen3-0.6B  (qwen3)
  architecture : 28 layers, hidden=1024, ...
  node-0      : layers 0:14 (14 blocks) port 127.0.0.1:31341
  node-1      : layers 14:28 (14 blocks) port 127.0.0.1:31342
[..] starting bootstrap on 127.0.0.1:31340
[ok] bootstrap: /ip4/127.0.0.1/tcp/31340/p2p/12D3Koo...
[..] starting node-0: layers 0:14, port 31341
[..] starting node-1: layers 14:28, port 31342
[..] waiting for 28 blocks to become ONLINE in the DHT
[ok] DHT ready: 2 server(s), 28 block(s) online
[..] running end-to-end smoke generation
[ok] smoke passed: 8 token(s); text='The capital of France is Paris...'

BloomBee local simulation is UP
```

The processes are detached (`start_new_session`), so they keep running after the
shell command exits. Stop them with:

```bash
./sim/simctl down
```

## Commands

| Command | Description |
|---|---|
| `./sim/simctl up` | Start bootstrap + workers, wait for DHT, run smoke test. Leaves the swarm running. |
| `./sim/simctl up --no-smoke` | Start without the generation check. |
| `./sim/simctl up --force` | Stop and restart an existing simulation. |
| `./sim/simctl status` | Show PIDs, ports, block ranges and alive/dead status. |
| `./sim/simctl logs [name]` | Tail `bootstrap`, `node-0`, ... logs (`--lines N`). |
| `./sim/simctl smoke` | Re-run the end-to-end generation check. |
| `./sim/simctl ask "..."` | Send a prompt to the running swarm and print the answer. |
| `./sim/simctl env` | Print `INITIAL_PEERS` / `MODEL_NAME` for the integration tests. |
| `./sim/simctl plan` | Show the block split and a VRAM estimate without starting anything. |
| `./sim/simctl doctor` | Check Python/Torch/CUDA/model config/ports. |
| `./sim/simctl down` | Stop all processes (state and logs are kept). |
| `./sim/simctl down --clean` | Stop and remove `sim/.state` + `sim/.logs` (HF cache is kept). |
| `./sim/simctl down --all` | Also kill orphaned BloomBee servers/clients not present in `run.json`. |
| `./sim/simctl restart` | `down` + `up`. |

Exit codes: `0` success, `1` configuration/environment error, `2` startup
failure, `3` smoke-test failure, `130` interrupted.

## Asking the model

With the swarm running:

```bash
./sim/simctl ask "Explain BloomBee in one sentence." --max-new-tokens 64
```

The command reads `sim/.state/run.json`, starts a client, applies the model's
chat template, sends the prompt through the DHT-routed pipeline, prints only the
answer, and exits. Useful flags:

| Flag | Meaning |
|---|---|
| `--max-new-tokens N` | generation budget (default 64) |
| `--temperature T` | `0` = greedy (default), `>0` = sampling |
| `--top-p P` | nucleus sampling when temperature > 0 |
| `--device cuda` | run the client (embeddings + LM head) on GPU; default CPU |
| `--dtype bfloat16` | client dtype override |
| `--raw` | plain completion instead of the chat template |
| `--thinking` | request Qwen3-style thinking mode |
| `--verbose` | show client warnings/routing logs |

Examples:

```bash
./sim/simctl ask "What is 2+2? Answer briefly." --max-new-tokens 16
./sim/simctl ask "Write a haiku about distributed systems." --temperature 0.7 --max-new-tokens 64
./sim/simctl ask "Continue: The capital of France is" --raw --max-new-tokens 8
```

For many requests, loading a fresh client per CLI call is wasteful. Reuse the
Python API directly:

```python
import torch
from transformers import AutoTokenizer
from bloombee import AutoDistributedModelForCausalLM

model = AutoDistributedModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    initial_peers=["/ip4/127.0.0.1/tcp/31340/p2p/<bootstrap-peer-id>"],
    dht_prefix="sim-Qwen3-0-6B",
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

messages = [{"role": "user", "content": "Explain BloomBee in one sentence."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64)

answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer)
```

Get the exact `initial_peers` and model id with `./sim/simctl env`.

> BloomBee is a client library, not an HTTP server: there is no REST/OpenAI
> endpoint. If you need `curl`/HTTP for the group demo, add a thin FastAPI
> wrapper around the client above.

## Container mode (isolated network)

A self-contained Docker image and an isolated bridge with `tc netem` are in
[`sim/docker/`](docker/README.md). The image includes Python, torch,
transformers, pinned Hivemind, BloomBee and the default model; runtime
containers need no host Python environment and no internet:

```bash
./sim/docker/run.sh up
./sim/docker/run.sh ask "Say hello in one sentence." --max-new-tokens 32
./sim/docker/run.sh down
```

## Manual block split

By default the 28 layers are split evenly between `--nodes`. To control the
split explicitly, use `start:end` ranges (end-exclusive) that must be contiguous
and cover every layer:

```bash
./sim/simctl up \
  --nodes 3 \
  --block-split 0:10,10:20,20:28 \
  --base-port 31400
```

The split is validated before anything starts: gaps, overlaps, out-of-range
indices, or a range count that differs from `--nodes` fail immediately.

## Configuration

Every option has an environment-variable default, so the whole stand can be
configured once (e.g. in `.envrc` or your shell profile):

| Variable | Default | Meaning |
|---|---|---|
| `SIM_MODEL` | `Qwen/Qwen3-0.6B` | HF model id or local path |
| `SIM_REVISION` | *(empty)* | Model revision/commit |
| `SIM_NODES` | `2` | Number of worker processes |
| `SIM_GPU` | `0` | `CUDA_VISIBLE_DEVICES` value (`0`, `1`, `MIG-...`) |
| `SIM_DEVICE` | `cuda` | `cuda` or `cpu` |
| `SIM_DTYPE` | `float32` | `float32`, `float16`, `bfloat16` |
| `SIM_CLIENT_DEVICE` | `cpu` | Where embeddings/LM head run (`cpu` saves VRAM) |
| `SIM_BASE_PORT` | `31340` | Bootstrap port; workers use `+1`, `+2`, ... |
| `SIM_BLOCK_SPLIT` | *(auto)* | e.g. `0:14,14:28` |
| `SIM_W_GPU_PERCENT` | `100` | FlexGen (LLaMA): % of block weights kept on GPU |
| `SIM_CACHE_GPU_PERCENT` | `100` | FlexGen: % of KV cache kept on GPU |
| `SIM_KV_TOKENS` | `512` | `--inference_max_length` / `--attn_cache_tokens` |
| `SIM_MAX_BATCH_SIZE` | `512` | Max tokens per server batch |
| `SIM_HANDLERS` | `1` | `--num_handlers` per worker |
| `SIM_CPU_THREADS` | `4` | `OMP_NUM_THREADS` / `MKL_NUM_THREADS` |
| `SIM_STATE_DIR` | `sim/.state` | PIDs, identities, per-node homes |
| `SIM_LOG_DIR` | `sim/.logs` | Worker/bootstrap/smoke logs |
| `SIM_HF_HOME` | `sim/.cache/hf` | Shared Hugging Face cache |
| `SIM_START_TIMEOUT` | `600` | Seconds to wait for DHT readiness |
| `SIM_SMOKE_TIMEOUT` | `300` | Seconds for the smoke generation |
| `SIM_PROMPT` | `The capital of France is` | Smoke prompt |
| `SIM_MAX_NEW_TOKENS` | `8` | Smoke generation length |

## Lifecycle

`up` is **not** a daemon and there is no supervisor. It performs one startup
transaction and then exits:

```
doctor + plan
  -> start bootstrap DHT          (CPU-only process)
  -> start N worker processes      (each serves its --block_indices range)
  -> wait until every block is ONLINE in the DHT
  -> start a client, generate a few tokens, stop the client
  -> leave bootstrap + workers running, write sim/.state/run.json
  -> exit 0
```

So after a successful `up` you have **N worker nodes plus one bootstrap peer**;
the client process is ephemeral and only exists during the smoke test. The
workers stay up until `./sim/simctl down`, `./sim/simctl up --force`, or an
explicit kill. Nothing restarts a worker if it crashes after `up` returns:
`status` will show it as dead, and `smoke` will fail.

Use `./sim/simctl up --no-smoke` to skip the generation check, and
`./sim/simctl up --keep-on-failure` to leave processes running when startup
fails. `./sim/simctl env` prints the `INITIAL_PEERS`/`MODEL_NAME` needed to run
the project's own pytest integration tests against the live swarm.

## What `up` verifies before saying PASS

1. Python/Torch/CUDA/transformers/hivemind/BloomBee import and version checks.
2. The model config is readable and the layer count is known.
3. The requested block split is contiguous, non-overlapping and covers all layers.
4. Bootstrap and worker ports are free.
5. Every worker process is still alive after startup.
6. The DHT reports an `ONLINE` server for every block `0..L-1`.
7. A real client loads embeddings/LM head through the DHT and generates tokens.

Any failure returns a non-zero exit code, prints the relevant log tail, and (by
default) stops the processes it started.

## How it works

```
sim/simctl.py
  ├── probe.py         -> torch/CUDA/versions (subprocess, no CUDA context in parent)
  ├── introspect.py    -> model layers/hidden size (subprocess)
  ├── run_dht          -> private bootstrap DHT on 127.0.0.1:31340
  ├── run_server x N   -> one process per contiguous block range
  │      HOME=<state>/node-i/home          (isolated FlexGen offload dir)
  │      TMPDIR=<state>/node-i/tmp         (isolated temp/weight-conversion dir)
  │      BLOOMBEE_CACHE=<hf-cache>/hub     (shared, protected by blocks.lock)
  │      HF_HOME=<hf-cache>                (weight download deduplicated)
  ├── wait_ready.py    -> polls DHT until every block has an ONLINE server
  └── smoke_client.py  -> loads embeddings + LM head, generates N tokens
```

Why each worker gets its own `HOME` and `TMPDIR`: BloomBee's FlexGen offload
directory is `~/.cache/bloombee/flexgen_offload_dir`, and FlexGen writes files
named `t_0`, `t_1`, ... using a process-local counter. Two workers sharing a
home directory can corrupt each other's offloaded tensors. Per-process `HOME`
and `TMPDIR` also isolate any temporary weight-conversion files. The Hugging
Face / block cache is shared on purpose: `blocks.lock` serialises eviction and
the HF hub cache deduplicates downloads between workers and the client.

Workers always receive an explicit `--block_indices`. BloomBee's automatic
`_choose_num_blocks()` uses the **total** GPU memory reported by
`torch.cuda.get_device_properties()`, not the free memory, so on a shared GPU it
would over-commit. Explicit ranges are also what makes the simulated topology
deterministic.

The readiness gate is DHT-based: `wait_ready.py` checks that every block
`0..L-1` has at least one server in `ServerState.ONLINE`. Only after that does
the smoke client run, so a "PASS" means the full client → DHT → workers →
client path works.

## Using the stand with the existing pytest suite

Most tests in `tests/` are integration tests that need a live swarm. Print the
exports:

```bash
./sim/simctl env
```

Then, for example:

```bash
INITIAL_PEERS='/ip4/.../p2p/...' MODEL_NAME='Qwen/Qwen3-0.6B' \
  pytest tests/test_full_model.py -v
```

Note that these tests may assume a specific model/size; check the test file
before running it against the small default model.

## Troubleshooting

### `CUDA out of memory` / workers die during startup

- Reduce the number of workers or use a smaller model.
- Use `--dtype bfloat16` (biggest win for HF-loaded models). `float16` also
  works, but then the client must run on GPU too (`--client-device cuda`).
- Add more nodes / reduce layers per node.
- Lower the KV budget: `--kv-tokens 256`.
- Run the client on CPU (default) so it does not compete for VRAM.
- `--w-gpu-percent 60 --cache-gpu-percent 60` helps only for FlexGen-based
  LLaMA blocks; Qwen3/Bloom/Falcon blocks are loaded by HF and keep their
  weights on the device, so the percentage flags do not offload them.
- Check `./sim/simctl logs node-i` — the OOM message is in the worker log.

### `port 31340 is already in use`

Another simulation or an unrelated service is using the base port. Use
`--base-port 31500` (workers use the next ports).

### `bloombee is not importable`

The interpreter chosen by `./sim/simctl` is not the project environment.
Activate the env and run `pip install -e .`, set
`PYTHON=/path/to/env/bin/python`, or use the prebuilt `sim/.venv` (the wrapper
automatically prefers it when present).

### `DHT readiness failed` / missing blocks

Inspect `sim/.logs/wait_ready.log` and the worker logs. Typical causes:

- a worker crashed while loading its block (OOM, missing file, wrong revision);
- the block split does not cover all layers;
- the model uses an architecture BloomBee does not support (the worker log says
  `BloomBee does not support model type ...`).

### Smoke test hangs or times out

- First run downloads weights; increase `SIM_SMOKE_TIMEOUT` if the network is
  slow.
- Check that the client can load the tokenizer (`HF_TOKEN` for gated models).
- Inspect `sim/.logs/smoke.log` and the worker logs for RPC errors.

### High RAM usage / `/dev/shm` is full

Two distinct causes have been observed:

1. **FlexGen copy-buffer over-allocation (fixed).** `TorchDisk` starts four copy
   workers, and each used to allocate its own 1 GiB pinned relay buffer. With
   CUDA allocation rounding that became ~8 GiB of host RSS per server and
   ~16 GiB for two local workers. The buffer is now
   `BLOOMBEE_COPY_BUFFER_MB` (default 64 MiB) per worker, and copies larger than
   the slab get a temporary pinned buffer. Reference measurement on an RTX 3070:
   two workers went from ~9.7 GiB RSS each to ~1.6 GiB, host `used` from 25 GiB
   to 9.6 GiB.
2. **Orphaned workers.** If a previous run was not stopped cleanly (or
   `run.json` was removed), orphaned workers can keep unlinked shared-memory
   segments alive. `ls /dev/shm` may look nearly empty while `free -h` still
   shows `Shmem` in the gigabytes.

```bash
./sim/simctl down --all
free -h
df -h /dev/shm
```

Then check for leftovers:

```bash
ps -eo pid,etime,cmd | grep -E 'bloombee|sim/(ask|smoke)_client' | grep -v grep
```

If RAM is still tight, lower the relay buffers:

```bash
BLOOMBEE_COPY_BUFFER_MB=16 ./sim/run.sh --kv-tokens 256 --max-batch-size 256
```

### Stale state after a crash

```bash
./sim/simctl down --clean
```

This removes per-node identities and logs but keeps the Hugging Face cache, so
restarting does not re-download the model.

## CPU-only mode (validated)

The same harness works without a GPU; it is slower but exercises the full
control plane (DHT, block routing, RPC, generation):

```bash
./sim/simctl up \
  --device cpu \
  --start-timeout 1800 \
  --smoke-timeout 1200 \
  --max-new-tokens 4
```

In CPU mode the harness automatically sets
`BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER=0` (no CUDA streams) and passes explicit
block ranges, so BloomBee never calls `_choose_num_blocks()`. FlexGen's copy
workers also need a CPU guard; see the `copy_worker_func` / `TorchTensor.synchronize`
handling in `src/bloombee/flexgen_utils/pytorch_backend.py`.

Reference run on this machine:

| Component | Version |
|---|---|
| Python | 3.11.18 |
| torch | 2.15.0+cpu |
| transformers | 5.21.0 |
| hivemind | 1.2.0.dev0 (commit `4bd43b7`) |
| peft | 0.8.2 |
| numpy | 1.26.4 |
| model | `Qwen/Qwen3-0.6B`, 28 layers, 2 workers × 14 |

Result: all 28 blocks `ONLINE`, smoke generation succeeded
(`The capital of France is Paris. Which is`).

## Limitations

- **One shared GPU.** Containers/processes do not virtualize CUDA: all workers
  see the same device and the same total VRAM. Memory is shared, not partitioned;
  a worker OOM can affect its neighbours. For hard isolation use MIG
  (A100/H100), separate GPUs, or a CUDA memory-fraction patch.
- **No network emulation yet.** All traffic goes over loopback. There is no
  latency/loss/bandwidth control. The next step is a Docker/netns profile with
  `tc netem` per worker; the `sim/` CLI is structured so this can be added
  without changing the worker configuration.
- **Performance numbers are not clean.** Workers contend for SMs and memory
  bandwidth, so throughput/latency measurements from this stand are indicative,
  not publication-quality. Use MIG or run scenarios sequentially for cleaner
  numbers.
- **First run is slow.** It downloads the model and builds per-node caches.
  Subsequent runs reuse `SIM_HF_HOME`.

## Files

```
sim/
├── simctl.py        orchestrator (stdlib only)
├── probe.py         runtime/CUDA/version probe
├── introspect.py    model config introspection
├── wait_ready.py    DHT readiness gate
├── smoke_client.py  end-to-end generation check
├── run.sh           one-click wrapper
└── README.md        this file
```
