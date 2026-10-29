#!/usr/bin/env bash
# One-click local BloomBee simulation: bootstrap + N workers + smoke test.
#
#   ./sim/run.sh
#   ./sim/run.sh --nodes 3 --block-split 0:10,10:20,20:28
#   PYTHON=/path/to/venv/bin/python ./sim/run.sh
#
# All arguments are forwarded to `sim/simctl up`; the wrapper picks the
# interpreter (PYTHON, then sim/.venv, then python/python3).
set -euo pipefail

cd "$(dirname "$0")/.."
exec "$PWD/sim/simctl" up "$@"
