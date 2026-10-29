#!/usr/bin/env python3
"""Emit a JSON snapshot of the local runtime (Python, Torch, CUDA, BloomBee).

This script is intentionally run as a *subprocess* by ``simctl.py``: importing
torch / bloombee creates a CUDA context, so the orchestrator must never do it
in its own process.  Always print exactly one JSON object on stdout.
"""

from __future__ import annotations

import json
import sys
import traceback


def _version(module, attr: str = "__version__") -> str:
    try:
        return str(getattr(module, attr))
    except Exception:
        return "unknown"


def main() -> int:
    result: dict = {"python": sys.version.split()[0]}

    try:
        import torch

        result["torch"] = _version(torch)
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
        gpus = []
        if result["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                try:
                    free, total = torch.cuda.mem_get_info(index)
                except Exception:
                    free, total = 0, props.total_memory
                gpus.append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_bytes": int(total),
                        "free_bytes": int(free),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
        result["gpus"] = gpus
    except Exception as exc:  # pragma: no cover - depends on the host env
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        result["torch_traceback"] = traceback.format_exc()

    try:
        import transformers

        result["transformers"] = _version(transformers)
    except Exception as exc:
        result["transformers_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import hivemind

        result["hivemind"] = _version(hivemind)
    except Exception as exc:
        result["hivemind_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import bloombee

        result["bloombee"] = _version(bloombee)
    except Exception as exc:
        result["bloombee_error"] = f"{type(exc).__name__}: {exc}"

    # Server-side optional import: missing peft only fails once a worker
    # constructs a TransformerBackend, so surface it during preflight.
    try:
        import peft

        result["peft"] = _version(peft)
    except Exception as exc:
        result["peft_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
