#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(dirname "$ROOT")"
RUNS="$WORKSPACE/mem0-runtime/stage1/runs"
mkdir -p -- "$RUNS"
RUN="$(mktemp -d "$RUNS/manual-demo-$(date +%Y%m%dT%H%M%S)-XXXXXX")"
PY="${STAGE1_PYTHON:-$WORKSPACE/mem0-runtime/conda/envs/mem0/bin/python}"
printf 'Stage1 synthetic run: %s\n' "$RUN"
# This demo never reads a key file or calls a model.
env -u DEEPSEEK_API_KEY -u OPENAI_API_KEY -u OPENROUTER_API_KEY \
  MEM0_TELEMETRY=False HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" "$ROOT/tools/stage1/demo.py" --run-dir "$RUN"
