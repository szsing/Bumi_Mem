#!/usr/bin/env bash
# 本次改动（2026-10-21）：提供独立目录、Agent、端口和 GPU 的隔离测试启动入口，默认 18111。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BUMI_SERVICE_ENV_FILE:?Set an external service configuration file}"
set -a
source "$BUMI_SERVICE_ENV_FILE"
set +a
export BUMI_DATA_DIR="${BUMI_ISOLATION_DATA_DIR:?Use a separate test data directory}"
export BUMI_AGENT_ID="${BUMI_ISOLATION_AGENT_ID:-bumi_isolation_console}"
export BUMI_HOST="${BUMI_ISOLATION_HOST:-127.0.0.1}"
export BUMI_PORT="${BUMI_ISOLATION_PORT:-18111}"
export BUMI_REGISTERED_ISOLATION=on
export BUMI_EXTRACTION_TURNS="${BUMI_ISOLATION_EXTRACTION_TURNS:-10}"
if [[ -n "${BUMI_ISOLATION_GPU:-}" ]]; then export CUDA_VISIBLE_DEVICES="$BUMI_ISOLATION_GPU"; fi
unset BUMI_SERVICE_ENV_FILE
exec bash "$ROOT/scripts/run_memory_service.sh"
