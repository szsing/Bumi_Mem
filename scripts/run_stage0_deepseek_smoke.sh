#!/usr/bin/env bash
set -euo pipefail
set +x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(dirname "$ROOT")"
PY="${STAGE0_PYTHON:-$WORKSPACE/mem0-runtime/conda/envs/mem0/bin/python}"
RUNS="$WORKSPACE/mem0-runtime/stage0/runs"
mkdir -p -- "$RUNS"
RUN="$(mktemp -d "$RUNS/api-smoke-$(date +%Y%m%dT%H%M%S)-XXXXXX")"
# Never source existing .env files, persist a key or put it in process arguments.
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then echo 'Run interactively to enter your key securely.' >&2; exit 2; fi
  read -r -s -p 'DeepSeek API Key (hidden; not saved): ' DEEPSEEK_API_KEY
  printf '\n'
fi
if [[ -z "$DEEPSEEK_API_KEY" ]]; then echo 'Empty key; no request sent.' >&2; exit 2; fi
export DEEPSEEK_API_KEY
trap 'unset DEEPSEEK_API_KEY' EXIT
printf 'One small billable API request; no private chat data. Report: %s/reports\n' "$RUN"
"$PY" "$ROOT/tools/stage0/stage0.py" --run-dir "$RUN" api-smoke \
  --model "${STAGE0_DEEPSEEK_MODEL:-deepseek-flash}"
