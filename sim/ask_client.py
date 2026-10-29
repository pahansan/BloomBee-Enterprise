#!/usr/bin/env python3
"""Send one prompt to a running BloomBee swarm and print the answer.

Called by the container runner (``./sim/docker/run.sh ask``) and by the
container entrypoint for the startup smoke test.

Output contract:
    stdout  the generated answer only (no logs)
    stderr  warnings / traceback on failure
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-cache-tokens", type=int, default=0, help="server KV capacity; 0 = no pre-check")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--raw", action="store_true", help="no chat template, plain completion")
    parser.add_argument("--thinking", action="store_true", help="ask Qwen3-style tokenizers for thinking mode")
    args = parser.parse_args()

    if args.device == "cpu" and args.dtype == "float16":
        print("float16 is not supported by the CPU client; use --dtype float32/bfloat16", file=sys.stderr)
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

        if args.raw:
            prompt_text = args.prompt
        else:
            messages = [{"role": "user", "content": args.prompt}]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(args.thinking),
                )
            except TypeError:
                # Tokenizers that do not know enable_thinking.
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        inputs = tokenizer(prompt_text, return_tensors="pt")
        if args.device == "cuda":
            inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        max_new_tokens = int(args.max_new_tokens)
        if args.max_cache_tokens > 0 and prompt_tokens + max_new_tokens > args.max_cache_tokens:
            needed = prompt_tokens + max_new_tokens
            print(
                f"ASK_FAIL: request needs {needed} tokens "
                f"(prompt {prompt_tokens} + max_new_tokens {max_new_tokens}), "
                f"but the running server KV cache is limited to {args.max_cache_tokens}. "
                f"Restart with --kv-tokens {needed} (and --max-batch-size >= {needed}) "
                f"or lower --max-new-tokens.",
                file=sys.stderr,
            )
            return 1

        generation_kwargs = {"max_new_tokens": max_new_tokens}
        if args.temperature and args.temperature > 0:
            generation_kwargs.update(
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)

        sequences = output.sequences if hasattr(output, "sequences") else output
        new_tokens = sequences[0][prompt_tokens:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(answer if answer else "(empty response)")
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"ASK_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
