#!/usr/bin/env python3
"""End-to-end smoke test for a locally running BloomBee swarm.

Loads the client-side modules (embeddings + LM head) through the DHT, generates a
few tokens through the remote transformer blocks, and prints a single
``SMOKE_OK {...}`` line on success.

Run via ``python sim/simctl.py smoke``; it is also called automatically by
``simctl.py up``.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial-peers", nargs="+", required=True)
    parser.add_argument("--dht-prefix", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype == "float16":
        print("SMOKE_FAIL: float16 is not supported by the CPU client; use --dtype float32", file=sys.stderr)
        return 1

    try:
        import torch
        from transformers import AutoTokenizer

        from bloombee import AutoDistributedModelForCausalLM

        dtype = getattr(torch, args.dtype)
        kwargs = {
            "initial_peers": args.initial_peers,
            "torch_dtype": dtype,
            "show_route": "inference",
        }
        if args.dht_prefix:
            kwargs["dht_prefix"] = args.dht_prefix
        if args.revision:
            kwargs["revision"] = args.revision

        model = AutoDistributedModelForCausalLM.from_pretrained(args.model, **kwargs)
        model.eval()
        if args.device == "cuda":
            model = model.to("cuda")

        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        inputs = tokenizer(args.prompt, return_tensors="pt")
        if args.device == "cuda":
            inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        sequences = output.sequences if hasattr(output, "sequences") else output
        prompt_tokens = int(inputs["input_ids"].shape[1])
        new_tokens = int(sequences.shape[1]) - prompt_tokens
        if new_tokens <= 0:
            print(f"SMOKE_FAIL: generated {new_tokens} new token(s)", file=sys.stderr)
            return 1

        text = tokenizer.decode(sequences[0], skip_special_tokens=True)
        print(
            "SMOKE_OK "
            + json.dumps(
                {
                    "prompt": args.prompt,
                    "new_tokens": new_tokens,
                    "text": text,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"SMOKE_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
