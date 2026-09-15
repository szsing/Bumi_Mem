#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
PYTHON="$MEM0_PYTHON"
SECRETS_FILE="${DEEPSEEK_ENV_FILE:-$MEM0_SECRETS_ROOT/deepseek.env}"

if [[ -f "$SECRETS_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
fi

export CHAT_PROVIDER="${CHAT_PROVIDER:-deepseek}"
export MEMORY_PROVIDER="${MEMORY_PROVIDER:-deepseek}"
export MEM0_EMBED_DEVICE="${MEM0_EMBED_DEVICE:-auto}"
export PYTHONPATH="$ROOT/src:$ROOT/src/mem0-new${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 Bumi_Mem Python 环境：$PYTHON" >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/apps/mem0_chat.py" "$@"
