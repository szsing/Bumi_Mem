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
exec "${BUMI_PYTHON:-python}" -m bumi_mem.api
