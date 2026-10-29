#!/usr/bin/env python3
"""Wait until every transformer block is served by an ONLINE peer in the DHT.

This is the readiness gate used by ``simctl.py up``.  It is a separate process so
that importing BloomBee/torch never pollutes the orchestrator process (and never
creates a CUDA context on the shared GPU).

Exit codes:
    0  all blocks have at least one ONLINE server; JSON summary on stdout
    1  timed out or DHT error; JSON diagnostics on stdout
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-peers", nargs="+", required=True)
    parser.add_argument("--dht-prefix", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    try:
        from bloombee.data_structures import UID_DELIMITER, ServerState
        from bloombee.utils.dht import get_remote_module_infos
        from bloombee.utils.hivemind_compat import DHT
    except Exception:
        traceback.print_exc()
        print(json.dumps({"ready": False, "error": "failed to import bloombee"}))
        return 1

    uids = [f"{args.dht_prefix}{UID_DELIMITER}{index}" for index in range(args.layers)]
    deadline = time.monotonic() + args.timeout
    last_report = 0.0
    last_missing = list(range(args.layers))
    dht = None

    try:
        while time.monotonic() < deadline and dht is None:
            try:
                dht = DHT(initial_peers=args.initial_peers, client_mode=True, start=True)
            except Exception as exc:
                print(f"[wait_ready] DHT client start failed (retrying): {exc}", file=sys.stderr)
                time.sleep(args.interval)
        if dht is None:
            print(json.dumps({"ready": False, "error": "could not start DHT client", "missing": last_missing}))
            return 1

        while time.monotonic() < deadline:
            try:
                infos = get_remote_module_infos(dht, uids, latest=True)
            except Exception as exc:
                now = time.monotonic()
                if now - last_report >= 5.0:
                    print(f"[wait_ready] DHT query failed (retrying): {exc}", file=sys.stderr)
                    last_report = now
                time.sleep(args.interval)
                continue

            online: dict[str, list[int]] = {}
            missing: list[int] = []
            for index, info in enumerate(infos):
                peers = [
                    str(peer_id)
                    for peer_id, server_info in info.servers.items()
                    if server_info.state == ServerState.ONLINE
                ]
                if not peers:
                    missing.append(index)
                for peer_id in peers:
                    online.setdefault(peer_id, []).append(index)

            last_missing = missing
            now = time.monotonic()
            if not missing:
                print(
                    json.dumps(
                        {
                            "ready": True,
                            "layers": args.layers,
                            "servers": online,
                            "server_count": len(online),
                        }
                    )
                )
                return 0

            if now - last_report >= 5.0:
                preview = ", ".join(str(index) for index in missing[:8])
                suffix = "..." if len(missing) > 8 else ""
                print(
                    f"[wait_ready] {args.layers - len(missing)}/{args.layers} blocks online; "
                    f"missing: {preview}{suffix}",
                    file=sys.stderr,
                )
                last_report = now

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(json.dumps({"ready": False, "error": "interrupted", "missing": last_missing}))
        return 1
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}", "missing": last_missing}))
        return 1
    finally:
        if dht is not None:
            try:
                dht.shutdown()
            except Exception:
                pass

    print(
        json.dumps(
            {
                "ready": False,
                "error": f"timed out after {args.timeout:.0f}s",
                "layers": args.layers,
                "missing": last_missing,
            }
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
