#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
VLLM="$VLLM_BIN"
MODEL="$MEM0_MODEL_ROOT/Qwen2.5-7B-Instruct"
HOST="127.0.0.1"
PORT="${VLLM_PORT:-8000}"
PID_FILE="$MEM0_TMP_ROOT/vllm-qwen2.5-7b.pid"
LOG_FILE="$MEM0_LOG_ROOT/vllm-qwen2.5-7b.log"

mkdir -p "$MEM0_TMP_ROOT" "$MEM0_LOG_ROOT"
if curl -fsS --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "vLLM 已在 http://$HOST:$PORT 运行。"
  exit 0
fi
if [[ ! -x "$VLLM" ]]; then
  echo "找不到 vLLM 运行环境：$VLLM" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "找不到模型：$MODEL" >&2
  exit 1
fi

cd "$ROOT"
export HOME="$MEM0_HOME_ROOT"
export XDG_CACHE_HOME="$MEM0_CACHE_ROOT"
export HF_HOME="$MEM0_CACHE_ROOT/huggingface"
export TMPDIR="$MEM0_TMP_ROOT"
export TOKENIZERS_PARALLELISM=false
nohup "$VLLM" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name qwen2.5-7b-instruct \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.75 \
  >>"$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"

for _ in $(seq 1 180); do
  if curl -fsS --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "vLLM 启动成功：pid=$PID url=http://$HOST:$PORT"
    echo "日志：$LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "vLLM 启动失败，请检查：$LOG_FILE" >&2
    exit 1
  fi
  sleep 1
done

rm -f "$PID_FILE"
echo "vLLM 在 180 秒内未就绪，请检查：$LOG_FILE" >&2
exit 1
