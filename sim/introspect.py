#!/usr/bin/env python3
"""Read a Hugging Face model config and print the fields the sim needs as JSON.

Deliberately does not import torch: the orchestrator only needs the architecture
size to split transformer blocks between local worker processes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _first_attr(obj, names, default=None):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1

    num_layers = _first_attr(config, ["num_hidden_layers", "n_layer", "num_layers"])
    hidden_size = _first_attr(config, ["hidden_size", "n_embd", "d_model"])
    if not num_layers or not hidden_size:
        print(
            json.dumps(
                {
                    "error": "could not determine num_hidden_layers/hidden_size "
                    f"for model_type={getattr(config, 'model_type', '?')}"
                }
            )
        )
        return 1

    intermediate_size = _first_attr(config, ["intermediate_size", "n_inner"], 4 * int(hidden_size))
    vocab_size = _first_attr(config, ["vocab_size"], 0)
    num_attention_heads = _first_attr(config, ["num_attention_heads", "n_head"], 0)
    num_key_value_heads = _first_attr(config, ["num_key_value_heads"], num_attention_heads)

    info = {
        "model": args.model,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "vocab_size": int(vocab_size),
        "num_attention_heads": int(num_attention_heads or 0),
        "num_key_value_heads": int(num_key_value_heads or 0),
        "max_position_embeddings": int(
            _first_attr(config, ["max_position_embeddings", "n_positions"], 0) or 0
        ),
        "torch_dtype": str(getattr(config, "torch_dtype", "unknown")),
    }
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
