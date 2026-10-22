#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BUMI_SERVICE_ENV_FILE:?Set an external service configuration file}"
set -a
source "$BUMI_SERVICE_ENV_FILE"
set +a
: "${BUMI_CONTROL_ENV_FILE:?Set a private control configuration file}"
set -a
source "$BUMI_CONTROL_ENV_FILE"
set +a
: "${BUMI_CONTROL_API_KEY:?Set a distinct control key of at least 32 characters}"
: "${BUMI_CONTROL_USER_ID:?Set the trusted test user ID}"
export BUMI_DATA_DIR="${BUMI_CONTROL_DATA_DIR:?Use a separate test data directory}"
export BUMI_AGENT_ID="${BUMI_CONTROL_AGENT_ID:-bumi_control_pilot}"
export BUMI_HOST="${BUMI_CONTROL_HOST:-127.0.0.1}"
export BUMI_PORT="${BUMI_CONTROL_PORT:-18110}"
if [[ -n "${BUMI_CONTROL_GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$BUMI_CONTROL_GPU"; fi
unset BUMI_SERVICE_ENV_FILE
exec bash "$ROOT/scripts/run_memory_service.sh"
