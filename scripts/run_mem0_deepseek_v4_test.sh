#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
SECRETS_FILE="${DEEPSEEK_ENV_FILE:-$MEM0_SECRETS_ROOT/deepseek.env}"
if [[ -f "$SECRETS_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
fi
export MEMORY_API_BASE_URL="${MEMORY_API_BASE_URL:-${OPENAI_BASE_URL:-https://api.deepseek.com/v1}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEEPSEEK_API_KEY:-}}"
if [[ -z "$OPENAI_API_KEY" ]]; then
  echo "缺少 DeepSeek API Key：请配置 $SECRETS_FILE" >&2
  exit 1
fi
exec "$ROOT/scripts/run_mem0_long_test.sh" \
  --memory-model "${DEEPSEEK_MODEL:-deepseek-v4-pro}" \
  --memory-base-url "$MEMORY_API_BASE_URL" \
  --memory-max-tokens "${MEM0_MEMORY_MAX_TOKENS:-4096}" \
  "$@"
