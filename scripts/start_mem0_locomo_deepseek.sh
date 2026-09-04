#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
PROJECT_NAME="${1:-locomo-deepseek-v2.0.20}"
if [[ $# -gt 0 ]]; then shift; fi
LOG="$MEM0_LOG_ROOT/locomo-launch-$PROJECT_NAME.log"
PID="$MEM0_TMP_ROOT/locomo-run-$PROJECT_NAME.pid"
mkdir -p "$MEM0_LOG_ROOT" "$MEM0_TMP_ROOT"
nohup "$ROOT/scripts/run_mem0_locomo_deepseek.sh" "$PROJECT_NAME" "$@" >"$LOG" 2>&1 &
echo $! > "$PID"
echo "LoCoMo 已在后台启动：PID $(cat "$PID")"
echo "启动日志：$LOG"
echo "查看进度：tail -f '$LOG'"
