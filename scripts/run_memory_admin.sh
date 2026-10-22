#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BUMI_SERVICE_ENV_FILE:?Set an external service configuration file}"
set -a
source "$BUMI_SERVICE_ENV_FILE"
set +a
export BUMI_DATA_DIR="${BUMI_ADMIN_DATA_DIR:-${BUMI_DATA_DIR:?}/admin-console}"
export BUMI_AGENT_ID="${BUMI_ADMIN_AGENT_ID:-bumi_admin_console}"
export BUMI_HOST="${BUMI_ADMIN_HOST:-127.0.0.1}"
export BUMI_PORT="${BUMI_ADMIN_PORT:-18109}"
if [[ -n "${BUMI_ADMIN_GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$BUMI_ADMIN_GPU"; fi
# Source once: the ordinary launcher must not overwrite administrator overrides.
unset BUMI_SERVICE_ENV_FILE
exec bash "$ROOT/scripts/run_memory_service.sh"
