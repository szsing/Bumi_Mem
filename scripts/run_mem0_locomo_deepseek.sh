#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
PYTHON="$MEM0_PYTHON"
BENCHMARK_ROOT="$ROOT/src/memory-benchmarks"
DATASET="${LOCOMO_DATASET:-/home/legendre/data/ZhixingSun/dataset/locomo10.json}"
PROJECT_NAME="${1:-locomo-deepseek-v2.0.20}"
if [[ $# -gt 0 ]]; then shift; fi

SECRETS_FILE="${DEEPSEEK_ENV_FILE:-$MEM0_SECRETS_ROOT/deepseek.env}"
if [[ -f "$SECRETS_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}"
if [[ -z "$OPENAI_API_KEY" ]]; then
  echo "缺少 DeepSeek API Key。请设置 OPENAI_API_KEY，或写入 $SECRETS_FILE" >&2
  exit 2
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${MEMORY_API_BASE_URL:-${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}}}"
export MEMORY_API_BASE_URL="$OPENAI_BASE_URL"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
export MEM0_REPRO_ROOT="$ROOT"
export MEM0_DATA_DIR="${MEM0_DATA_DIR:-$MEM0_DATA_ROOT/locomo_v3/$PROJECT_NAME}"
export MEM0_COLLECTION_NAME="${MEM0_COLLECTION_NAME:-locomo_v3_memories}"
export MEM0_EMBED_MODEL="${MEM0_EMBED_MODEL:-$MEM0_MODEL_ROOT/Qwen3-Embedding-0.6B}"
export MEM0_EMBED_DEVICE="${MEM0_EMBED_DEVICE:-cpu}"
export MEM0_MEMORY_MAX_TOKENS="${MEM0_MEMORY_MAX_TOKENS:-4096}"
export MEM0_TELEMETRY=False
export TOKENIZERS_PARALLELISM=false
export HOME="$MEM0_HOME_ROOT"
export XDG_CACHE_HOME="$MEM0_CACHE_ROOT"
export HF_HOME="$MEM0_CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$MEM0_CACHE_ROOT/huggingface/hub"
export PIP_CACHE_DIR="$MEM0_CACHE_ROOT/pip"
export TMPDIR="$MEM0_TMP_ROOT"

HOST="127.0.0.1"
PORT="${MEM0_LOCOMO_PORT:-8888}"
SERVER_LOG="$MEM0_LOG_ROOT/locomo-server-$PROJECT_NAME.log"
RUN_LOG="$MEM0_LOG_ROOT/locomo-run-$PROJECT_NAME.log"
OUTPUT_DIR="$MEM0_RESULTS_ROOT/locomo"
PID_FILE="$MEM0_TMP_ROOT/locomo-server-$PROJECT_NAME.pid"
mkdir -p "$MEM0_DATA_DIR" "$OUTPUT_DIR" "$MEM0_LOG_ROOT" "$MEM0_TMP_ROOT" "$MEM0_SECRETS_ROOT"
chmod 700 "$MEM0_SECRETS_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 MemNew Python 环境：$PYTHON" >&2
  exit 1
fi
if [[ ! -f "$DATASET" ]]; then
  echo "找不到 LoCoMo 数据集：$DATASET" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

"$PYTHON" -m uvicorn apps.mem0_locomo_server:app \
  --app-dir "$ROOT" --host "$HOST" --port "$PORT" --workers 1 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$PID_FILE"

for _ in $(seq 1 60); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Mem0 服务启动失败，请检查：$SERVER_LOG" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS "http://$HOST:$PORT/health" || {
  echo "Mem0 服务未能在 60 秒内就绪，请检查：$SERVER_LOG" >&2
  exit 1
}
echo

echo "开始 LoCoMo：project=$PROJECT_NAME model=$DEEPSEEK_MODEL top_k=${LOCOMO_TOP_K:-200}"
echo "服务日志：$SERVER_LOG"
echo "运行日志：$RUN_LOG"

cd "$BENCHMARK_ROOT"
set +e
"$PYTHON" -m benchmarks.locomo.run \
  --project-name "$PROJECT_NAME" \
  --dataset-path "$DATASET" \
  --backend oss \
  --mem0-host "http://$HOST:$PORT" \
  --answerer-model "$DEEPSEEK_MODEL" \
  --judge-model "$DEEPSEEK_MODEL" \
  --provider openai \
  --top-k "${LOCOMO_TOP_K:-200}" \
  --top-k-cutoffs "${LOCOMO_TOP_K_CUTOFFS:-200}" \
  --max-workers "${LOCOMO_MAX_WORKERS:-2}" \
  --rpm "${LOCOMO_RPM:-120}" \
  --output-dir "$OUTPUT_DIR" \
  "$@" 2>&1 | tee "$RUN_LOG"
STATUS=${PIPESTATUS[0]}
set -e
exit "$STATUS"
