#!/usr/bin/env python3
"""One-click local BloomBee swarm on a single machine / single GPU.

The orchestrator is deliberately stdlib-only: it never imports torch or BloomBee
itself, because that would create a CUDA context in the parent process and steal
memory from the workers.  All heavy lifting happens in short-lived subprocesses:

    probe.py        runtime / CUDA / version snapshot
    introspect.py   model config -> number of layers, hidden size, ...
    wait_ready.py   poll the DHT until every block has an ONLINE server
    smoke_client.py end-to-end generation through the local swarm

Typical usage:

    ./sim/simctl up
    ./sim/simctl status
    ./sim/simctl logs node-0
    ./sim/simctl smoke
    ./sim/simctl down

Or just:  ./sim/run.sh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

SIM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SIM_DIR.parent
STATE_FILE = "run.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_BASE_PORT = 31340
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


class SimError(RuntimeError):
    """User-facing error: print a concise message, not a traceback."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


@dataclass
class SimConfig:
    model: str
    revision: Optional[str]
    nodes: int
    gpu: str
    device: str
    dtype: str
    client_device: str
    client_dtype: str
    base_port: int
    block_split: Optional[str]
    dht_prefix: str
    w_gpu_percent: int
    cache_gpu_percent: int
    kv_tokens: int
    max_batch_size: int
    handlers: int
    cpu_threads: int
    state_dir: Path
    log_dir: Path
    hf_home: Path
    start_timeout: float
    smoke_timeout: float
    prompt: str
    max_new_tokens: int

    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        data = dict(data)
        for key in ("state_dir", "log_dir", "hf_home"):
            data[key] = Path(data[key])
        return cls(**data)


def _dht_prefix_for(model: str) -> str:
    """Stable DHT prefix, independent of the default derivation in each config."""
    name = Path(model.rstrip("/")).name or "model"
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-") or "model"
    return f"sim-{name}"


def build_config(args) -> SimConfig:
    model = str(args.model).strip()
    if not model:
        raise SimError("--model must not be empty")

    device = args.device
    gpu = str(args.gpu).strip()
    if device == "cuda" and not gpu:
        raise SimError("--gpu is required for CUDA mode (e.g. --gpu 0 or --gpu MIG-...)")

    client_device = args.client_device
    if device == "cpu":
        client_device = "cpu"
    if client_device == "cpu" and args.dtype == "float16":
        raise SimError(
            "float16 is not usable by a CPU client; use --dtype float32/bfloat16 "
            "or --client-device cuda"
        )

    nodes = int(args.nodes)
    if nodes < 1:
        raise SimError("--nodes must be >= 1")

    base_port = int(args.base_port)
    if not (1 <= base_port <= 65535 - nodes - 1):
        raise SimError(f"--base-port {base_port} does not leave room for {nodes} worker ports")

    for name, value in (("--w-gpu-percent", args.w_gpu_percent), ("--cache-gpu-percent", args.cache_gpu_percent)):
        if not 0 <= int(value) <= 100:
            raise SimError(f"{name} must be in [0, 100], got {value}")

    for name, value in (
        ("--kv-tokens", args.kv_tokens),
        ("--max-batch-size", args.max_batch_size),
        ("--handlers", args.handlers),
        ("--cpu-threads", args.cpu_threads),
    ):
        if int(value) < 1:
            raise SimError(f"{name} must be >= 1, got {value}")
    if int(args.max_batch_size) < int(args.kv_tokens):
        raise SimError("--max-batch-size must be >= --kv-tokens")

    state_dir = Path(args.state_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    hf_home = Path(args.hf_home).expanduser().resolve()

    revision = (args.revision or "").strip() or None
    block_split = (args.block_split or "").strip() or None

    return SimConfig(
        model=model,
        revision=revision,
        nodes=nodes,
        gpu=gpu,
        device=device,
        dtype=args.dtype,
        client_device=client_device,
        client_dtype=args.dtype,
        base_port=base_port,
        block_split=block_split,
        dht_prefix=_dht_prefix_for(model),
        w_gpu_percent=int(args.w_gpu_percent),
        cache_gpu_percent=int(args.cache_gpu_percent),
        kv_tokens=int(args.kv_tokens),
        max_batch_size=int(args.max_batch_size),
        handlers=int(args.handlers),
        cpu_threads=int(args.cpu_threads),
        state_dir=state_dir,
        log_dir=log_dir,
        hf_home=hf_home,
        start_timeout=float(args.start_timeout),
        smoke_timeout=float(args.smoke_timeout),
        prompt=str(args.prompt),
        max_new_tokens=int(args.max_new_tokens),
    )


# --------------------------------------------------------------------------- #
# process / file helpers
# --------------------------------------------------------------------------- #


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            # state is the third field; "Z" means a zombie waiting to be reaped
            return stat.read_text(errors="replace").split()[2] != "Z"
        except Exception:
            pass
    return True


def _pid_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        return ""


_BLOOMBEE_CMDLINE_MARKERS = (
    "bloombee.cli.run_dht",
    "bloombee.cli.run_server",
    "sim/ask_client.py",
    "sim/smoke_client.py",
)


def _is_bloombee_process(pid: int) -> bool:
    if not Path("/proc").exists():
        return True  # non-Linux: trust the pid file
    cmdline = _pid_cmdline(pid)
    return any(marker in cmdline for marker in _BLOOMBEE_CMDLINE_MARKERS)


def _scan_orphaned_services() -> list[dict]:
    """Find BloomBee servers/clients that are not referenced by run.json."""
    found = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return found
    current_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid or not _pid_alive(pid):
            continue
        cmdline = _pid_cmdline(pid)
        if any(marker in cmdline for marker in _BLOOMBEE_CMDLINE_MARKERS):
            found.append({"name": f"orphan-{pid}", "pid": pid, "log": ""})
    return found


def _base_env(cfg: SimConfig) -> dict:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache = cfg.hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": str(cfg.hf_home),
            "HUGGINGFACE_HUB_CACHE": str(hub_cache),
            "TRANSFORMERS_CACHE": str(hub_cache),
            # BloomBee uses this directory both as its HF cache and as the block
            # cache guarded by blocks.lock; sharing it deduplicates downloads
            # between workers and the client.
            "BLOOMBEE_CACHE": str(hub_cache),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(cfg.cpu_threads),
            "MKL_NUM_THREADS": str(cfg.cpu_threads),
            "CUDA_VISIBLE_DEVICES": cfg.gpu if cfg.device == "cuda" else "",
        }
    )
    if cfg.revision:
        env["SIM_REVISION"] = cfg.revision
    if cfg.device == "cpu":
        # No CUDA streams/events in a CPU-only PyTorch build.
        env["BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER"] = "0"
        env["BLOOMBEE_ENABLE_MICROBATCH_PIPELINE"] = "0"
    return env


def _node_envs(cfg: SimConfig, base_env: dict, index: int) -> tuple[dict, Path]:
    node_dir = cfg.state_dir / f"node-{index}"
    home = node_dir / "home"
    tmp = node_dir / "tmp"
    for path in (home, tmp):
        path.mkdir(parents=True, exist_ok=True)

    env = dict(base_env)
    # FlexGen's offload directory is derived from HOME, and some Llama weight
    # conversion code writes under TMPDIR. Both must be per-worker; the HF/block
    # cache stays shared and is protected by blocks.lock.
    env["HOME"] = str(home)
    env["TMPDIR"] = str(tmp)
    return env, node_dir



def _save_state(cfg: SimConfig, state: dict) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.state_dir / STATE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_state(cfg: SimConfig) -> Optional[dict]:
    path = cfg.state_dir / STATE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SimError(f"cannot read {path}: {exc}") from exc


def _paths_from_args(args) -> tuple[Path, Path]:
    return (
        Path(args.state_dir).expanduser().resolve(),
        Path(args.log_dir).expanduser().resolve(),
    )


def _remove_state(cfg: SimConfig) -> None:
    try:
        (cfg.state_dir / STATE_FILE).unlink()
    except FileNotFoundError:
        pass


def _tail(path: Path, lines: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines))
    except FileNotFoundError:
        return ""


def _parse_last_json(text: str) -> Optional[dict]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _run_json(cmd: list[str], env: dict, timeout: float, what: str) -> dict:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SimError(f"{what} timed out after {timeout:.0f}s") from exc

    payload = _parse_last_json(proc.stdout)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        raise SimError(f"{what} failed (exit {proc.returncode}):\n{detail}")
    if payload is None:
        detail = (proc.stdout + "\n" + proc.stderr).strip()[-3000:]
        raise SimError(f"{what} produced no JSON output:\n{detail}")
    return payload


def _spawn(name: str, cmd: list[str], env: dict, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc


def _kill_group(pid: int, sig: int) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def _services(state: dict) -> list[dict]:
    services = []
    if state.get("bootstrap"):
        services.append(state["bootstrap"])
    services.extend(state.get("nodes", []))
    return services


def _alive_services(state: dict) -> list[dict]:
    alive = []
    for service in _services(state):
        pid = int(service.get("pid") or 0)
        if _pid_alive(pid) and _is_bloombee_process(pid):
            alive.append(service)
    return alive


def _stop_services(state: dict) -> None:
    services = list(reversed(_services(state)))
    to_kill = []
    for service in services:
        pid = int(service.get("pid") or 0)
        if not _pid_alive(pid):
            continue
        if not _is_bloombee_process(pid):
            print(
                f"[warn] refusing to kill pid {pid} ({service.get('name')}): "
                f"unexpected cmdline {_pid_cmdline(pid)[:160]!r}",
                file=sys.stderr,
            )
            continue
        to_kill.append(service)

    for service in to_kill:
        _kill_group(int(service["pid"]), signal.SIGTERM)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(
        _pid_alive(int(service["pid"])) for service in to_kill
    ):
        time.sleep(0.2)

    for service in to_kill:
        pid = int(service["pid"])
        if _pid_alive(pid):
            print(f"[warn] force-killing pid {pid} ({service.get('name')})", file=sys.stderr)
            _kill_group(pid, signal.SIGKILL)

    for service in to_kill:
        print(f"[ok] stopped {service.get('name')} (pid {service['pid']})")


def _safe_rmtree(path: Path) -> None:
    path = path.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), REPO_ROOT.resolve()}
    if path in forbidden:
        raise SimError(f"refusing to remove {path}")
    if not path.exists():
        return
    shutil.rmtree(path)


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def compute_ranges(total_layers: int, nodes: int, split: Optional[str]) -> list[tuple[int, int]]:
    """Return [(start, end), ...] with end-exclusive contiguous ranges covering all layers."""
    if total_layers < 1:
        raise SimError("model has no transformer layers")
    if split:
        ranges: list[tuple[int, int]] = []
        for chunk in split.split(","):
            match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", chunk)
            if not match:
                raise SimError(f"invalid --block-split entry {chunk!r}; expected start:end")
            start, end = int(match.group(1)), int(match.group(2))
            if not 0 <= start < end <= total_layers:
                raise SimError(
                    f"block range {start}:{end} is outside [0, {total_layers}] or empty"
                )
            ranges.append((start, end))
        if len(ranges) != nodes:
            raise SimError(f"--block-split defines {len(ranges)} ranges, but --nodes is {nodes}")
        expected = 0
        for start, end in ranges:
            if start != expected:
                raise SimError(
                    f"--block-split must be contiguous and cover all layers; expected start {expected}, got {start}"
                )
            expected = end
        if expected != total_layers:
            raise SimError(f"--block-split ends at {expected}, but the model has {total_layers} layers")
        return ranges

    if nodes > total_layers:
        raise SimError(
            f"cannot split {total_layers} layers between {nodes} nodes; reduce --nodes"
        )
    base, remainder = divmod(total_layers, nodes)
    ranges = []
    start = 0
    for index in range(nodes):
        size = base + (1 if index < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def estimate_resources(info: dict, cfg: SimConfig, ranges: list[tuple[int, int]]) -> dict:
    hidden = int(info["hidden_size"])
    intermediate = int(info["intermediate_size"])
    layers = int(info["num_hidden_layers"])
    vocab = int(info.get("vocab_size") or 0)
    bytes_per_param = DTYPE_BYTES[cfg.dtype]

    # Upper-bound estimate for Llama-like blocks: q/k/v/o + gate/up/down.
    per_layer_params = 4 * hidden * hidden + 3 * hidden * intermediate
    total_params = per_layer_params * layers + 2 * vocab * hidden
    max_node_layers = max(end - start for start, end in ranges)
    node_params = per_layer_params * max_node_layers
    kv_per_layer = 2 * hidden * cfg.kv_tokens * bytes_per_param
    node_bytes = node_params * bytes_per_param + kv_per_layer * max_node_layers + 512 * 1024**2
    total_bytes = total_params * bytes_per_param
    client_bytes = 2 * vocab * hidden * bytes_per_param

    gib = 1024**3
    return {
        "total_params": total_params,
        "total_gib": total_bytes / gib,
        "max_node_gib": node_bytes / gib,
        "client_gib": client_bytes / gib,
        "kv_per_layer_gib": kv_per_layer / gib,
        "max_node_layers": max_node_layers,
    }


def _fmt_gib(value: float) -> str:
    return f"{value:.2f} GiB"


def _print_plan(cfg: SimConfig, info: dict, ranges: list[tuple[int, int]], probe: Optional[dict] = None) -> None:
    estimate = estimate_resources(info, cfg, ranges)
    print("Plan")
    print(f"  model        : {cfg.model}  ({info.get('model_type')})")
    print(
        f"  architecture : {info['num_hidden_layers']} layers, "
        f"hidden={info['hidden_size']}, intermediate={info['intermediate_size']}, "
        f"vocab={info.get('vocab_size')}"
    )
    print(f"  dtype        : {cfg.dtype}")
    print(f"  dht prefix   : {cfg.dht_prefix}")
    print(f"  bootstrap    : 127.0.0.1:{cfg.base_port}")
    for index, (start, end) in enumerate(ranges):
        print(
            f"  node-{index:<7}: layers {start}:{end} ({end - start} blocks) "
            f"port 127.0.0.1:{cfg.base_port + 1 + index}"
        )
    print(
        f"  estimate     : total ~{_fmt_gib(estimate['total_gib'])} "
        f"({estimate['total_params'] / 1e9:.2f}B params); "
        f"max worker ~{_fmt_gib(estimate['max_node_gib'])} incl. KV+context"
    )
    if probe and probe.get("gpus"):
        gpu = probe["gpus"][0]
        free_gib = gpu["free_bytes"] / 1024**3
        total_gib = gpu["total_bytes"] / 1024**3
        print(
            f"  gpu          : {gpu['name']} ({free_gib:.2f} / {total_gib:.2f} GiB free), "
            f"CC {gpu['compute_capability']}"
        )
        if free_gib < estimate["max_node_gib"] * 1.3:
            print(
                "  [warn] workers may not fit: lower --w-gpu-percent/--cache-gpu-percent "
                "or use more nodes / a smaller model"
            )
    elif cfg.device == "cuda":
        print("  gpu          : no CUDA device visible to the probe")
    else:
        print("  device       : CPU-only mode (slow, intended for control-plane tests)")


def _check_ports(cfg: SimConfig) -> None:
    ports = [cfg.base_port] + [cfg.base_port + 1 + index for index in range(cfg.nodes)]
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SimError(
                f"port {port} is already in use ({exc}); "
                "stop the other process or pass --base-port <free port>"
            ) from exc
        finally:
            sock.close()


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value or "")
    return tuple(int(part) for part in parts[:3])


def _doctor(
    cfg: SimConfig,
    quiet: bool = False,
    check_ports: bool = True,
) -> tuple[dict, dict, list[tuple[int, int]]]:
    base_env = _base_env(cfg)

    probe_env = dict(base_env)
    if cfg.device == "cpu":
        probe_env["CUDA_VISIBLE_DEVICES"] = ""

    probe = _run_json(
        [sys.executable, str(SIM_DIR / "probe.py")],
        env=probe_env,
        timeout=120,
        what="runtime probe",
    )
    if not quiet:
        print("Environment")
        print(f"  python       : {probe.get('python')}")
        print(f"  torch        : {probe.get('torch', 'missing')}")
        print(f"  transformers : {probe.get('transformers', 'missing')}")
        print(f"  hivemind     : {probe.get('hivemind', 'missing')}")
        print(f"  bloombee     : {probe.get('bloombee', 'missing')}")

    if probe.get("torch_error"):
        raise SimError(
            "PyTorch is not importable:\n"
            f"{probe['torch_error']}\n"
            "Activate the project environment and run: pip install -e ."
        )
    if probe.get("transformers_error"):
        raise SimError(f"transformers is not importable: {probe['transformers_error']}")

    transformers_version = str(probe.get("transformers", "0"))
    if _version_tuple(transformers_version) < (5, 5, 0):
        raise SimError(
            f"transformers>=5.5.0 is required, found {transformers_version}; "
            "run: pip install -U 'transformers>=5.5.0'"
        )

    if probe.get("hivemind_error"):
        raise SimError(f"hivemind is not importable: {probe['hivemind_error']}")
    if probe.get("bloombee_error"):
        raise SimError(
            "bloombee is not importable from this interpreter:\n"
            f"{probe['bloombee_error']}\n"
            "Install the package in the active environment: pip install -e ."
        )
    if probe.get("peft_error"):
        raise SimError(
            "peft is not importable, but BloomBee's server imports it while creating a block:\n"
            f"{probe['peft_error']}\n"
            "Install it in the active environment: pip install -e ."
        )

    if cfg.device == "cuda":
        if not probe.get("cuda_available"):
            raise SimError(
                "CUDA is not available to PyTorch. Check nvidia-smi, the driver, and that "
                "torch was installed with a matching CUDA wheel. For a CPU-only smoke test, "
                "pass --device cpu."
            )
        gpus = probe.get("gpus") or []
        if not gpus:
            raise SimError("CUDA is reported available but no GPU properties were returned")
        free_gib = gpus[0]["free_bytes"] / 1024**3
        if free_gib < 1.5:
            raise SimError(
                f"only {free_gib:.2f} GiB of GPU memory is free; free some memory or use --device cpu"
            )
    else:
        print("  [ok] CPU-only mode: CUDA is hidden from workers")

    introspect_env = dict(base_env)
    introspect_env["CUDA_VISIBLE_DEVICES"] = ""
    info = _run_json(
        [
            sys.executable,
            str(SIM_DIR / "introspect.py"),
            "--model",
            cfg.model,
            *(["--revision", cfg.revision] if cfg.revision else []),
        ],
        env=introspect_env,
        timeout=180,
        what="model config introspection",
    )
    if info.get("error"):
        raise SimError(f"cannot read model config for {cfg.model}: {info['error']}")

    ranges = compute_ranges(int(info["num_hidden_layers"]), cfg.nodes, cfg.block_split)
    if check_ports:
        _check_ports(cfg)

    try:
        disk = shutil.disk_usage(cfg.hf_home)
        if disk.free < 5 * 1024**3:
            print(
                f"  [warn] only {disk.free / 1024**3:.1f} GiB free under {cfg.hf_home}; "
                "model download may fail"
            )
    except OSError:
        pass

    _print_plan(cfg, info, ranges, probe)
    return probe, info, ranges


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def _wait_for_bootstrap(proc: subprocess.Popen, log_path: Path, timeout: float) -> str:
    pattern = re.compile(r"(/ip4/[0-9.]+/tcp/\d+/p2p/[A-Za-z0-9]+)")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        addresses = pattern.findall(text)
        if addresses:
            for address in addresses:
                if address.startswith("/ip4/127.0.0.1"):
                    return address
            return addresses[0]
        if proc.poll() is not None:
            raise SimError(
                f"bootstrap exited during startup (code {proc.returncode}).\n"
                f"{_tail(log_path, 40)}"
            )
        time.sleep(0.5)
    raise SimError(
        f"bootstrap did not announce a peer address within {timeout:.0f}s.\n"
        f"{_tail(log_path, 40)}"
    )


def _start_bootstrap(cfg: SimConfig, base_env: dict) -> tuple[subprocess.Popen, dict]:
    boot_dir = cfg.state_dir / "bootstrap"
    home = boot_dir / "home"
    tmp = boot_dir / "tmp"
    for path in (home, tmp):
        path.mkdir(parents=True, exist_ok=True)

    env = dict(base_env)
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    port = cfg.base_port
    log_path = cfg.log_dir / "bootstrap.log"
    cmd = [
        sys.executable,
        "-m",
        "bloombee.cli.run_dht",
        "--host_maddrs",
        f"/ip4/127.0.0.1/tcp/{port}",
        "--announce_maddrs",
        f"/ip4/127.0.0.1/tcp/{port}",
        "--identity_path",
        str(boot_dir / "identity.id"),
        "--no_relay",
        "--refresh_period",
        "10",
    ]
    print(f"[..] starting bootstrap on 127.0.0.1:{port}")
    proc = _spawn("bootstrap", cmd, env, log_path)
    address = _wait_for_bootstrap(proc, log_path, timeout=90)
    print(f"[ok] bootstrap: {address}")

    service = {
        "name": "bootstrap",
        "kind": "dht",
        "pid": proc.pid,
        "port": port,
        "log": str(log_path),
        "addr": address,
        "proc": proc,
    }
    return proc, service


def _start_node(
    cfg: SimConfig,
    base_env: dict,
    index: int,
    start: int,
    end: int,
    address: str,
) -> tuple[subprocess.Popen, dict]:
    env, node_dir = _node_envs(cfg, base_env, index)
    port = cfg.base_port + 1 + index
    log_path = cfg.log_dir / f"node-{index}.log"
    cmd = [
        sys.executable,
        "-m",
        "bloombee.cli.run_server",
        cfg.model,
        "--initial_peers",
        address,
        "--identity_path",
        str(node_dir / "identity.id"),
        "--host_maddrs",
        f"/ip4/127.0.0.1/tcp/{port}",
        "--announce_maddrs",
        f"/ip4/127.0.0.1/tcp/{port}",
        "--block_indices",
        f"{start}:{end}",
        "--dht_prefix",
        cfg.dht_prefix,
        "--device",
        "cuda:0" if cfg.device == "cuda" else "cpu",
        "--torch_dtype",
        cfg.dtype,
        "--num_handlers",
        str(cfg.handlers),
        "--prefetch_batches",
        "1",
        "--sender_threads",
        "1",
        "--max_batch_size",
        str(cfg.max_batch_size),
        "--inference_max_length",
        str(cfg.kv_tokens),
        "--attn_cache_tokens",
        str(cfg.kv_tokens),
        "--max_alloc_timeout",
        "60",
        "--w_gpu_percent",
        str(cfg.w_gpu_percent),
        "--cache_gpu_percent",
        str(cfg.cache_gpu_percent),
        "--update_period",
        "15",
        "--expiration",
        "60",
        "--request_timeout",
        "120",
        "--session_timeout",
        "600",
        "--step_timeout",
        "120",
        "--throughput",
        "1.0",
        "--no_auto_relay",
        "--skip_reachability_check",
    ]
    if cfg.revision:
        cmd += ["--revision", cfg.revision]

    print(f"[..] starting node-{index}: layers {start}:{end}, port {port}")
    proc = _spawn(f"node-{index}", cmd, env, log_path)
    service = {
        "name": f"node-{index}",
        "kind": "server",
        "pid": proc.pid,
        "port": port,
        "blocks": [start, end],
        "log": str(log_path),
        "home": str(node_dir),
        "proc": proc,
    }
    return proc, service


def _wait_for_ready(
    cfg: SimConfig,
    address: str,
    layers: int,
    node_procs: list[tuple[str, subprocess.Popen, Path]],
) -> dict:
    base_env = _base_env(cfg)
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        sys.executable,
        str(SIM_DIR / "wait_ready.py"),
        "--initial-peers",
        address,
        "--dht-prefix",
        cfg.dht_prefix,
        "--layers",
        str(layers),
        "--timeout",
        str(cfg.start_timeout),
    ]
    log_path = cfg.log_dir / "wait_ready.log"
    print(f"[..] waiting for {layers} blocks to become ONLINE in the DHT")
    proc = _spawn("wait_ready", cmd, env, log_path)
    deadline = time.monotonic() + cfg.start_timeout + 30
    try:
        while proc.poll() is None:
            if time.monotonic() > deadline:
                _kill_group(proc.pid, signal.SIGKILL)
                raise SimError(
                    f"readiness check timed out after {cfg.start_timeout:.0f}s.\n"
                    f"{_tail(log_path, 40)}"
                )
            for name, node_proc, node_log in node_procs:
                if node_proc.poll() is not None:
                    _kill_group(proc.pid, signal.SIGKILL)
                    raise SimError(
                        f"{name} exited during startup (code {node_proc.returncode}).\n"
                        f"--- {node_log} ---\n{_tail(node_log, 60)}"
                    )
            time.sleep(1.0)

        output = _tail(log_path, 200)
        payload = _parse_last_json(output)
        if proc.returncode != 0 or not payload or not payload.get("ready"):
            missing = payload.get("missing") if payload else None
            hint = f" missing blocks: {missing}" if missing else ""
            raise SimError(
                f"DHT readiness failed (exit {proc.returncode}).{hint}\n"
                f"--- {log_path} ---\n{output[-4000:]}"
            )
        print(
            f"[ok] DHT ready: {payload.get('server_count', '?')} server(s), "
            f"{payload.get('layers')} block(s) online"
        )
        # ONLINE is announced when the module container exists; give the handler
        # and runtime processes a moment to start accepting RPCs.
        time.sleep(1.0)
        return payload
    finally:
        if proc.poll() is None:
            _kill_group(proc.pid, signal.SIGKILL)


def _run_smoke(cfg: SimConfig, address: str) -> bool:
    env = _base_env(cfg)
    client_dir = cfg.state_dir / "client"
    for sub in (client_dir / "home", client_dir / "tmp"):
        sub.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(client_dir / "home"),
            "TMPDIR": str(client_dir / "tmp"),
        }
    )
    if cfg.client_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        sys.executable,
        str(SIM_DIR / "smoke_client.py"),
        "--model",
        cfg.model,
        "--initial-peers",
        address,
        "--dht-prefix",
        cfg.dht_prefix,
        "--device",
        cfg.client_device,
        "--dtype",
        cfg.client_dtype,
        "--prompt",
        cfg.prompt,
        "--max-new-tokens",
        str(cfg.max_new_tokens),
    ]
    if cfg.revision:
        cmd += ["--revision", cfg.revision]

    print("[..] running end-to-end smoke generation")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=cfg.smoke_timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        output = (output or "") + f"\nSMOKE_FAIL: timed out after {cfg.smoke_timeout:.0f}s"
        proc.returncode = 124

    (cfg.log_dir / "smoke.log").write_text(output or "", encoding="utf-8")
    ok_line = next((line for line in (output or "").splitlines() if line.startswith("SMOKE_OK ")), None)
    if proc.returncode == 0 and ok_line:
        try:
            payload = json.loads(ok_line[len("SMOKE_OK ") :])
            text = str(payload.get("text", "")).replace("\n", " ")
            print(f"[ok] smoke passed: {payload.get('new_tokens')} token(s); text={text[:160]!r}")
        except json.JSONDecodeError:
            print(f"[ok] smoke passed: {ok_line}")
        return True

    print("[fail] smoke generation failed; last lines:", file=sys.stderr)
    tail = "\n".join((output or "").splitlines()[-40:])
    print(tail, file=sys.stderr)
    print(f"[fail] full smoke log: {cfg.log_dir / 'smoke.log'}", file=sys.stderr)
    return False


def _print_failure_logs(state: Optional[dict], cfg: SimConfig) -> None:
    print("\n--- logs ---", file=sys.stderr)
    services = _services(state) if state else []
    if not services:
        for log_path in sorted(cfg.log_dir.glob("*.log")):
            print(f"==> {log_path} <==", file=sys.stderr)
            print(_tail(log_path, 30), file=sys.stderr)
        return
    for service in services:
        log_path = Path(service.get("log", ""))
        print(f"==> {service.get('name')} ({log_path}) <==", file=sys.stderr)
        print(_tail(log_path, 40), file=sys.stderr)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_up(args) -> int:
    cfg = build_config(args)
    state = _load_state(cfg)

    if state:
        alive = _alive_services(state)
        if alive and not args.force:
            print(f"Simulation already appears to be running ({len(alive)} process(es)).")
            print("Use --force to restart, or './sim/simctl down' first.")
            if args.no_smoke:
                return 0
            address = state.get("bootstrap", {}).get("addr")
            if not address:
                raise SimError("state file exists but bootstrap address is missing; run down --clean")
            running_cfg = SimConfig.from_dict(state["config"])
            return 0 if _run_smoke(running_cfg, address) else 3
        if alive and args.force:
            print("[..] --force: stopping the existing simulation")
            _stop_services(state)
            still_alive = _alive_services(state)
            if still_alive:
                raise SimError(
                    "could not stop existing processes: "
                    + ", ".join(f"{s['name']} (pid {s['pid']})" for s in still_alive)
                )
        _remove_state(cfg)

    probe, info, ranges = _doctor(cfg)
    base_env = _base_env(cfg)

    state = {
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "starting",
        "model": cfg.model,
        "config": cfg.to_dict(),
        "bootstrap": None,
        "nodes": [],
        "smoke": None,
    }
    _save_state(cfg, state)

    try:
        _, bootstrap = _start_bootstrap(cfg, base_env)
        state["bootstrap"] = {key: value for key, value in bootstrap.items() if key != "proc"}
        _save_state(cfg, state)
        address = bootstrap["addr"]

        node_procs = []
        for index, (start, end) in enumerate(ranges):
            proc, service = _start_node(cfg, base_env, index, start, end, address)
            state["nodes"].append({key: value for key, value in service.items() if key != "proc"})
            _save_state(cfg, state)
            node_procs.append((service["name"], proc, Path(service["log"])))

        _wait_for_ready(cfg, address, int(info["num_hidden_layers"]), node_procs)
        state["status"] = "up"
        _save_state(cfg, state)

        if args.no_smoke:
            print("[ok] smoke skipped (--no-smoke)")
            state["smoke"] = "skipped"
            _save_state(cfg, state)
        else:
            if not _run_smoke(cfg, address):
                state["status"] = "failed"
                state["smoke"] = "failed"
                _save_state(cfg, state)
                _print_failure_logs(state, cfg)
                if not args.keep_on_failure:
                    _stop_services(state)
                    print("[..] stopped all processes after failure; logs are kept", file=sys.stderr)
                return 3
            state["smoke"] = "passed"
            _save_state(cfg, state)

        print()
        print("BloomBee local simulation is UP")
        print(f"  model      : {cfg.model}")
        print(f"  bootstrap  : {address}")
        for service in state["nodes"]:
            start, end = service["blocks"]
            print(
                f"  {service['name']:<10} layers {start}:{end}  port {service['port']}  "
                f"pid {service['pid']}  log {service['log']}"
            )
        print(f"  smoke      : {state['smoke']}")
        print()
        print("Next:")
        print("  ./sim/simctl status")
        print("  ./sim/simctl smoke")
        print("  ./sim/simctl logs node-0")
        print("  ./sim/simctl env        # INITIAL_PEERS / MODEL_NAME for pytest")
        print("  ./sim/simctl down")
        return 0

    except KeyboardInterrupt:
        print("\n[..] interrupted, stopping processes", file=sys.stderr)
        _print_failure_logs(state, cfg)
        _stop_services(state)
        return 130
    except SimError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        _print_failure_logs(state, cfg)
        if state and not args.keep_on_failure:
            _stop_services(state)
            print("[..] stopped all processes after failure; logs are kept", file=sys.stderr)
        return 2
    except Exception as exc:  # unexpected bug in the orchestrator
        print(f"\nUNEXPECTED FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        _print_failure_logs(state, cfg)
        if state and not args.keep_on_failure:
            _stop_services(state)
        return 2


def cmd_down(args) -> int:
    state_dir, log_dir = _paths_from_args(args)
    cfg = _standalone_cfg(state_dir, log_dir)
    state = _load_state(cfg)
    if state:
        _stop_services(state)
        _remove_state(cfg)
    else:
        print("No running simulation state found.")

    if getattr(args, "all", False):
        orphans = _scan_orphaned_services()
        if orphans:
            print(f"[..] stopping {len(orphans)} orphaned BloomBee process(es)")
            _stop_services({"nodes": orphans})
        else:
            print("[ok] no orphaned BloomBee processes found")

    if args.clean:
        for path in (state_dir, log_dir):
            _safe_rmtree(path)
        print(f"[ok] removed {state_dir} and {log_dir} (HF cache was kept)")
    else:
        print(f"[ok] state and logs kept under {state_dir} and {log_dir}")
    return 0


def _standalone_cfg(state_dir: Path, log_dir: Path) -> SimConfig:
    """Minimal config for commands that only need paths."""
    return SimConfig(
        model="",
        revision=None,
        nodes=0,
        gpu="",
        device="cpu",
        dtype="float32",
        client_device="cpu",
        client_dtype="float32",
        base_port=0,
        block_split=None,
        dht_prefix="",
        w_gpu_percent=100,
        cache_gpu_percent=100,
        kv_tokens=0,
        max_batch_size=0,
        handlers=1,
        cpu_threads=1,
        state_dir=state_dir,
        log_dir=log_dir,
        hf_home=SIM_DIR / ".cache" / "hf",
        start_timeout=0,
        smoke_timeout=0,
        prompt="",
        max_new_tokens=0,
    )


def cmd_status(args) -> int:
    state_dir, log_dir = _paths_from_args(args)
    cfg = _standalone_cfg(state_dir, log_dir)
    state = _load_state(cfg)
    if not state:
        print("Simulation is not running (no state file).")
        return 0

    alive = {
        service["name"]: _pid_alive(int(service.get("pid") or 0)) and _is_bloombee_process(int(service.get("pid") or 0))
        for service in _services(state)
    }
    if args.json:
        print(json.dumps({**state, "alive": alive}, indent=2, ensure_ascii=False))
        return 0

    state_label = state.get("status", "?").upper()
    if alive and not any(alive.values()):
        state_label = "STALE (all processes are dead)"
    print(f"Simulation: {state_label}  since {state.get('created_at', '?')}")
    print(f"Model     : {state.get('model')}")
    print(f"DHT prefix: {state.get('config', {}).get('dht_prefix')}")
    if state.get("smoke"):
        print(f"Smoke     : {state['smoke']}")
    print()
    print(f"{'service':<12} {'pid':>7} {'status':<8} {'port':>6}  blocks")
    for service in _services(state):
        pid = int(service.get("pid") or 0)
        status = "alive" if alive.get(service["name"]) else "dead"
        blocks = service.get("blocks")
        block_text = f"{blocks[0]}:{blocks[1]}" if blocks else "-"
        print(f"{service['name']:<12} {pid:>7} {status:<8} {service.get('port', '-'):>6}  {block_text}")
    if state.get("bootstrap", {}).get("addr"):
        print(f"\nbootstrap: {state['bootstrap']['addr']}")
    print(f"logs:      {log_dir}")
    return 0


def cmd_logs(args) -> int:
    state_dir, log_dir = _paths_from_args(args)
    cfg = _standalone_cfg(state_dir, log_dir)
    state = _load_state(cfg)
    names = []
    if args.name:
        names = [args.name]
    elif state:
        names = [service["name"] for service in _services(state)]
    else:
        names = [path.stem for path in sorted(log_dir.glob("*.log"))]

    if not names:
        raise SimError(f"no logs found under {log_dir}")

    for name in names:
        path = log_dir / f"{name}.log"
        if not path.exists():
            raise SimError(f"no such log: {path}")
        text = _tail(path, args.lines)
        print(f"==> {name} ({path}) <==")
        print(text, end="" if text.endswith("\n") else "\n")
        print()
    return 0


def cmd_smoke(args) -> int:
    state_dir, log_dir = _paths_from_args(args)
    cfg = _standalone_cfg(state_dir, log_dir)
    state = _load_state(cfg)
    if not state:
        raise SimError("simulation is not running; start it with './sim/simctl up'")
    if not _alive_services(state):
        raise SimError("simulation state exists, but no processes are alive; run './sim/simctl down' and 'up'")

    cfg = SimConfig.from_dict(state["config"])
    if args.timeout is not None:
        cfg.smoke_timeout = float(args.timeout)
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = int(args.max_new_tokens)
    if args.prompt is not None:
        cfg.prompt = str(args.prompt)

    address = state.get("bootstrap", {}).get("addr")
    if not address:
        raise SimError("state file has no bootstrap address; run down --clean and up again")
    return 0 if _run_smoke(cfg, address) else 3


def cmd_ask(args) -> int:
    """Send one prompt to the running swarm; answer goes to stdout."""
    state_dir, log_dir = _paths_from_args(args)
    cfg = _standalone_cfg(state_dir, log_dir)
    state = _load_state(cfg)
    if not state:
        raise SimError("simulation is not running; start it with './sim/simctl up'")
    if not _alive_services(state):
        raise SimError("simulation state exists, but no processes are alive; run './sim/simctl up --force'")

    cfg = SimConfig.from_dict(state["config"])
    if args.device:
        cfg.client_device = args.device
    if args.dtype:
        cfg.client_dtype = args.dtype
    if cfg.client_device == "cpu" and cfg.client_dtype == "float16":
        raise SimError("float16 is not usable by a CPU client; use --dtype float32/bfloat16 or --device cuda")

    address = state.get("bootstrap", {}).get("addr")
    if not address:
        raise SimError("state file has no bootstrap address; run './sim/simctl down --clean' and up again")

    env = _base_env(cfg)
    client_dir = cfg.state_dir / "client"
    for sub in (client_dir / "home", client_dir / "tmp"):
        sub.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(client_dir / "home"),
            "TMPDIR": str(client_dir / "tmp"),
        }
    )
    if cfg.client_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    prompt = " ".join(args.prompt)
    cmd = [
        sys.executable,
        str(SIM_DIR / "ask_client.py"),
        "--model",
        cfg.model,
        "--initial-peers",
        address,
        "--device",
        cfg.client_device,
        "--dtype",
        cfg.client_dtype,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--max-cache-tokens",
        str(int(cfg.kv_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--top-p",
        str(float(args.top_p)),
    ]
    if cfg.dht_prefix:
        cmd += ["--dht-prefix", cfg.dht_prefix]
    if cfg.revision:
        cmd += ["--revision", cfg.revision]
    if args.raw:
        cmd.append("--raw")
    if args.thinking:
        cmd.append("--thinking")

    print(f"[..] asking {cfg.model} (client={cfg.client_device}, max_new_tokens={args.max_new_tokens})", file=sys.stderr)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=float(args.timeout))
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        out = out or ""
        err = (err or "") + f"\nASK_FAIL: timed out after {float(args.timeout):.0f}s"
        proc.returncode = 124

    (cfg.log_dir / "ask.log").write_text(
        (out or "") + "\n--- stderr ---\n" + (err or ""), encoding="utf-8"
    )
    if proc.returncode != 0:
        print("[fail] request failed; see sim/.logs/ask.log", file=sys.stderr)
        if out:
            print(out, file=sys.stderr)
        if err:
            print(err, file=sys.stderr)
        return 3

    print((out or "").strip())
    if args.verbose and err:
        print(err, file=sys.stderr)
    return 0


def cmd_env(args) -> int:
    state_dir, log_dir = _paths_from_args(args)
    cfg = _standalone_cfg(state_dir, log_dir)
    state = _load_state(cfg)
    if not state:
        raise SimError("simulation is not running; start it with './sim/simctl up'")
    address = state.get("bootstrap", {}).get("addr")
    if not address:
        raise SimError("state file has no bootstrap address")

    cfg = SimConfig.from_dict(state["config"])
    payload = {
        "INITIAL_PEERS": address,
        "MODEL_NAME": cfg.model,
        "HF_HOME": str(cfg.hf_home),
        "DHT_PREFIX": cfg.dht_prefix,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for key, value in payload.items():
        print(f"export {key}={value!r}")
    print()
    print("# Example integration test against this local swarm:")
    print(
        f"INITIAL_PEERS={address!r} MODEL_NAME={cfg.model!r} "
        f"{sys.executable} -m pytest tests/test_full_model.py -v"
    )
    return 0


def cmd_plan(args) -> int:
    cfg = build_config(args)
    base_env = _base_env(cfg)
    probe_env = dict(base_env)
    if cfg.device == "cpu":
        probe_env["CUDA_VISIBLE_DEVICES"] = ""
    probe = None
    try:
        probe = _run_json([sys.executable, str(SIM_DIR / "probe.py")], probe_env, 120, "runtime probe")
    except SimError as exc:
        print(f"[warn] runtime probe failed: {exc}", file=sys.stderr)

    introspect_env = dict(base_env)
    introspect_env["CUDA_VISIBLE_DEVICES"] = ""
    info = _run_json(
        [
            sys.executable,
            str(SIM_DIR / "introspect.py"),
            "--model",
            cfg.model,
            *(["--revision", cfg.revision] if cfg.revision else []),
        ],
        introspect_env,
        180,
        "model config introspection",
    )
    if info.get("error"):
        raise SimError(f"cannot read model config for {cfg.model}: {info['error']}")
    ranges = compute_ranges(int(info["num_hidden_layers"]), cfg.nodes, cfg.block_split)
    try:
        _check_ports(cfg)
    except SimError as exc:
        print(f"[warn] {exc}", file=sys.stderr)
    _print_plan(cfg, info, ranges, probe)
    print("\nNo processes were started.")
    return 0


def cmd_doctor(args) -> int:
    cfg = build_config(args)
    _doctor(cfg, check_ports=False)
    try:
        _check_ports(cfg)
        print("[ok] all worker ports are free")
    except SimError as exc:
        print(f"[warn] {exc}", file=sys.stderr)
        print("[warn] ports above may be used by an already running simulation")
    print("\n[ok] environment looks usable")
    return 0


def cmd_restart(args) -> int:
    down_args = argparse.Namespace(
        state_dir=args.state_dir,
        log_dir=args.log_dir,
        clean=False,
    )
    cmd_down(down_args)
    return cmd_up(args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=_env("SIM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--revision", default=_env("SIM_REVISION", ""))
    parser.add_argument("--nodes", type=int, default=_env_int("SIM_NODES", 2))
    parser.add_argument("--gpu", default=_env("SIM_GPU", "0"), help="CUDA_VISIBLE_DEVICES value (0, 1, MIG-...)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=_env("SIM_DEVICE", "cuda"))
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=_env("SIM_DTYPE", "float32"))
    parser.add_argument(
        "--client-device",
        choices=["cpu", "cuda"],
        default=_env("SIM_CLIENT_DEVICE", "cpu"),
        help="where client-side embeddings/LM head run (CPU keeps VRAM for workers)",
    )
    parser.add_argument("--base-port", type=int, default=_env_int("SIM_BASE_PORT", DEFAULT_BASE_PORT))
    parser.add_argument("--block-split", default=_env("SIM_BLOCK_SPLIT", ""), help='e.g. "0:14,14:28"')
    parser.add_argument("--w-gpu-percent", type=int, default=_env_int("SIM_W_GPU_PERCENT", 100))
    parser.add_argument("--cache-gpu-percent", type=int, default=_env_int("SIM_CACHE_GPU_PERCENT", 100))
    parser.add_argument("--kv-tokens", type=int, default=_env_int("SIM_KV_TOKENS", 512))
    parser.add_argument("--max-batch-size", type=int, default=_env_int("SIM_MAX_BATCH_SIZE", 512))
    parser.add_argument("--handlers", type=int, default=_env_int("SIM_HANDLERS", 1))
    parser.add_argument("--cpu-threads", type=int, default=_env_int("SIM_CPU_THREADS", 4))
    parser.add_argument("--state-dir", default=_env("SIM_STATE_DIR", str(SIM_DIR / ".state")))
    parser.add_argument("--log-dir", default=_env("SIM_LOG_DIR", str(SIM_DIR / ".logs")))
    parser.add_argument("--hf-home", default=_env("SIM_HF_HOME", str(SIM_DIR / ".cache" / "hf")))
    parser.add_argument("--start-timeout", type=float, default=_env_float("SIM_START_TIMEOUT", 600.0))
    parser.add_argument("--smoke-timeout", type=float, default=_env_float("SIM_SMOKE_TIMEOUT", 300.0))
    parser.add_argument("--prompt", default=_env("SIM_PROMPT", "The capital of France is"))
    parser.add_argument("--max-new-tokens", type=int, default=_env_int("SIM_MAX_NEW_TOKENS", 8))


def _add_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dir", default=_env("SIM_STATE_DIR", str(SIM_DIR / ".state")))
    parser.add_argument("--log-dir", default=_env("SIM_LOG_DIR", str(SIM_DIR / ".logs")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single-device BloomBee swarm and verify it end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="start bootstrap + workers and run a smoke test")
    _add_model_args(up)
    up.add_argument("--force", action="store_true", help="restart an already running simulation")
    up.add_argument("--no-smoke", action="store_true", help="skip the end-to-end smoke test")
    up.add_argument("--keep-on-failure", action="store_true", help="leave processes running if startup fails")
    up.set_defaults(func=cmd_up)

    restart = sub.add_parser("restart", help="down + up")
    _add_model_args(restart)
    restart.add_argument("--force", action="store_true")
    restart.add_argument("--no-smoke", action="store_true")
    restart.add_argument("--keep-on-failure", action="store_true")
    restart.set_defaults(func=cmd_restart)

    down = sub.add_parser("down", help="stop all simulation processes")
    _add_path_args(down)
    down.add_argument("--clean", action="store_true", help="also remove state and logs (HF cache is kept)")
    down.add_argument("--all", action="store_true", help="also kill orphaned BloomBee servers/clients not tracked in run.json")
    down.set_defaults(func=cmd_down)

    status = sub.add_parser("status", help="show simulation status")
    _add_path_args(status)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="print worker/bootstrap logs")
    _add_path_args(logs)
    logs.add_argument("name", nargs="?", help="bootstrap, node-0, node-1, ... (default: all)")
    logs.add_argument("--lines", type=int, default=80)
    logs.set_defaults(func=cmd_logs)

    smoke = sub.add_parser("smoke", help="run the end-to-end smoke test against a running simulation")
    _add_path_args(smoke)
    smoke.add_argument("--timeout", type=float, default=None)
    smoke.add_argument("--max-new-tokens", type=int, default=None)
    smoke.add_argument("--prompt", default=None)
    smoke.set_defaults(func=cmd_smoke)

    ask = sub.add_parser("ask", help="send a prompt to the running swarm and print the answer")
    _add_path_args(ask)
    ask.add_argument("prompt", nargs="+", help="question / instruction text")
    ask.add_argument("--max-new-tokens", type=int, default=64)
    ask.add_argument("--temperature", type=float, default=0.0)
    ask.add_argument("--top-p", type=float, default=0.9)
    ask.add_argument("--device", choices=["cpu", "cuda"], default=None, help="override client device")
    ask.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    ask.add_argument("--raw", action="store_true", help="plain completion instead of a chat template")
    ask.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode in the chat template")
    ask.add_argument("--timeout", type=float, default=600.0)
    ask.add_argument("--verbose", action="store_true")
    ask.set_defaults(func=cmd_ask)

    env = sub.add_parser("env", help="print INITIAL_PEERS/MODEL_NAME for pytest")
    _add_path_args(env)
    env.add_argument("--json", action="store_true")
    env.set_defaults(func=cmd_env)

    plan = sub.add_parser("plan", help="show the block split and VRAM estimate without starting anything")
    _add_model_args(plan)
    plan.set_defaults(func=cmd_plan)

    doctor = sub.add_parser("doctor", help="check the local environment")
    _add_model_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SimError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
