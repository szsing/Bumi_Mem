#!/usr/bin/env bash
# Qdrant Server 启停/状态/日志/轮转。所有路径与端口可用 QDRANT_* 环境变量覆盖。
#
# 用法：
#   scripts/qdrant_server.sh start|stop|restart|status|logs|rotate
#
# 默认值（与本机 H20 部署一致）：
#   QDRANT_BIN=~/.local/opt/qdrant-musl-1.19.1/qdrant
#   QDRANT_STORAGE=$MEM0_RUNTIME_ROOT/qdrant-server/storage
#   QDRANT_LOG=$MEM0_RUNTIME_ROOT/logs/qdrant-server.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${MEM0_WORKSPACE_ROOT:-$(cd "$ROOT/../.." && pwd)}"
RUNTIME="${MEM0_RUNTIME_ROOT:-$WORKSPACE/mem0-runtime}"

QDRANT_BIN="${QDRANT_BIN:-$WORKSPACE/.local/opt/qdrant-musl-1.19.1/qdrant}"
STORAGE="${QDRANT_STORAGE:-$RUNTIME/qdrant-server/storage}"
LOG="${QDRANT_LOG:-$RUNTIME/logs/qdrant-server.log}"
PID_FILE="${QDRANT_PID_FILE:-$RUNTIME/qdrant-server/qdrant.pid}"
HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
LOG_MAX_BYTES="${QDRANT_LOG_MAX_BYTES:-52428800}"   # 50MB
LOG_KEEP="${QDRANT_LOG_KEEP:-3}"

is_up() { curl -sf --max-time 2 "http://127.0.0.1:$HTTP_PORT/healthz" >/dev/null 2>&1; }

rotate_log() {
  [[ -f "$LOG" ]] || return 0
  local size
  size="$(stat -c%s "$LOG" 2>/dev/null || echo 0)"
  if (( size > LOG_MAX_BYTES )); then
    for (( i=LOG_KEEP-1; i>=1; i-- )); do
      [[ -f "$LOG.$i" ]] && mv "$LOG.$i" "$LOG.$((i+1))"
    done
    # copytruncate：先复制再原地截断，运行中的 Qdrant 仍写同一个 inode，不会丢后续日志。
    cp "$LOG" "$LOG.1"
    : > "$LOG"
    echo "日志已轮转（copytruncate）：$LOG -> $LOG.1 (${size} bytes, 保留 $LOG_KEEP 份)"
  fi
}

start() {
  if is_up; then
    echo "Qdrant 已在运行（:$HTTP_PORT）"
    return 0
  fi
  if [[ ! -x "$QDRANT_BIN" ]]; then
    echo "找不到可执行的 Qdrant：$QDRANT_BIN" >&2
    echo "可从 https://github.com/qdrant/qdrant/releases 下载 musl 静态版后设置 QDRANT_BIN" >&2
    exit 1
  fi
  mkdir -p "$STORAGE" "$(dirname "$LOG")" "$(dirname "$PID_FILE")"
  # 启动时按大小轮转一次；常驻环境可加 cron/systemd timer 定期执行 `qdrant_server.sh rotate`。
  rotate_log
  # 切到 storage 的父目录再启动：Qdrant 会在工作目录写 .qdrant-initialized，
  # 不切换就会把标记文件丢在调用者的当前目录（例如仓库根）。
  cd "$(dirname "$STORAGE")"
  QDRANT__STORAGE__STORAGE_PATH="$STORAGE" \
  QDRANT__SERVICE__HTTP_PORT="$HTTP_PORT" \
  QDRANT__SERVICE__GRPC_PORT="$GRPC_PORT" \
  QDRANT__LOG_LEVEL="${QDRANT_LOG_LEVEL:-warn}" \
    setsid nohup "$QDRANT_BIN" >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  for _ in $(seq 1 30); do
    if is_up; then
      echo "Qdrant 启动成功：pid=$(cat "$PID_FILE") http=:$HTTP_PORT grpc=:$GRPC_PORT"
      echo "storage=$STORAGE"
      return 0
    fi
    sleep 1
  done
  echo "Qdrant 启动超时，检查日志：$LOG" >&2
  exit 1
}

stop() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    local pid; pid="$(cat "$PID_FILE")"
    kill "$pid"
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "pid=$pid 未在 30s 内退出，请手动检查" >&2
      exit 1
    fi
    rm -f "$PID_FILE"
    echo "Qdrant 已停止：pid=$pid"
    return 0
  fi
  if is_up; then
    echo "端口 $HTTP_PORT 上仍有 Qdrant 在运行，但没有本文档管理的 pid 文件；未做任何操作" >&2
    exit 1
  fi
  echo "Qdrant 未在运行"
}

status() {
  local pid="-"
  [[ -f "$PID_FILE" ]] && pid="$(cat "$PID_FILE")"
  if is_up; then
    echo "running  http=:$HTTP_PORT grpc=:$GRPC_PORT pid=$pid"
    curl -sf --max-time 3 "http://127.0.0.1:$HTTP_PORT/collections" \
      | python3 -c 'import json,sys; print("collections:", [c["name"] for c in json.load(sys.stdin)["result"]["collections"]])' \
      2>/dev/null || true
  else
    echo "stopped  (pid file: ${pid})"
    exit 3
  fi
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) tail -n "${2:-50}" "$LOG" ;;
  rotate) rotate_log ;;
  *)
    echo "用法：$0 {start|stop|restart|status|logs [行数]|rotate}" >&2
    exit 2
    ;;
esac
