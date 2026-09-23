#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Deliberately do not source the legacy runtime_env.sh (it rewrites HOME/paths).
if [[ -n "${BUMI_SERVICE_ENV_FILE:-}" ]]; then
  if [[ ! -f "$BUMI_SERVICE_ENV_FILE" ]]; then
    echo "BUMI_SERVICE_ENV_FILE does not exist" >&2
    exit 2
  fi
  set -a
  source "$BUMI_SERVICE_ENV_FILE"
  set +a
fi
export PYTHONPATH="$ROOT/src:$ROOT/src/mem0-new${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

# 没配 BUMI_PYTHON 时按常见布局找一个可用的 venv，避免直接 exec 到不存在的 python。
PYTHON="${BUMI_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  WORKSPACE="${MEM0_WORKSPACE_ROOT:-$(cd "$ROOT/.." && pwd)}"
  for candidate in \
    "$WORKSPACE/.venvs/bumi-cuda/bin/python" \
    "$WORKSPACE/.venvs/bumi/bin/python" \
    "$ROOT/../../.venvs/bumi-cuda/bin/python" \
    "$ROOT/../../.venvs/bumi/bin/python" \
    "$ROOT/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "找不到可用的 Python 环境：请在 service.env 里设置 BUMI_PYTHON" >&2
  exit 1
fi
echo "使用 Python：$PYTHON" >&2
exec "$PYTHON" -m bumi_mem.api
