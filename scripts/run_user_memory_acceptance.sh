#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(dirname "$ROOT")"
RUNS="$WORKSPACE/mem0-runtime/acceptance/user_memory/runs"
mkdir -p -- "$RUNS"
RUN="$(mktemp -d "$RUNS/check-$(date +%Y%m%dT%H%M%S)-XXXXXX")"
PY="${MEM0_PYTHON:-$WORKSPACE/mem0-runtime/conda/envs/mem0/bin/python}"
printf 'Isolated acceptance run: %s\n' "$RUN"
exec "$PY" "$ROOT/tools/user_memory/acceptance.py" run --run-dir "$RUN"
