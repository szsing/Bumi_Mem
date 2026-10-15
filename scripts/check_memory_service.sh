#!/usr/bin/env bash
# One small billable LLM probe by default; use --skip-llm for offline checks only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${BUMI_SERVICE_ENV_FILE:?Set BUMI_SERVICE_ENV_FILE to the new Graphiti configuration}"
[[ -f "$ENV_FILE" ]] || { echo "Missing configuration: $ENV_FILE" >&2; exit 2; }
set -a
source "$ENV_FILE"
set +a
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "${BUMI_PYTHON:-python}" -m bumi_mem.service_probe "$@"
