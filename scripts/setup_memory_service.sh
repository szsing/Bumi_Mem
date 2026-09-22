#!/usr/bin/env bash
# Interactive first-time configuration; no keys are shipped in this repository.
set +x
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${BUMI_RUNTIME_DIR:-/data1/sunzhixing/bumi-memory-service}"
MODEL_DIR="${BUMI_EMBED_MODEL_PATH:-/data1/sunzhixing/model/Qwen3-Embedding-0.6B}"
PYTHON="${BUMI_PYTHON:-python}"
[[ -d "$MODEL_DIR" ]] || { echo "Embedding directory does not exist: $MODEL_DIR" >&2; exit 2; }
[[ -t 0 ]] || { echo 'Run this script in your server terminal; API keys must not be sent in chat or command arguments.' >&2; exit 2; }
PYTHON_PATH="$("$PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required"; print(sys.executable)')"
"$PYTHON_PATH" - "$APP_DIR" "$RUNTIME_DIR" <<'PY'
import sys
from pathlib import Path
app, runtime = (Path(p).expanduser().resolve() for p in sys.argv[1:])
if runtime == app or app in runtime.parents:
    raise SystemExit('Keep runtime data and secrets OUTSIDE the Git repository')
PY
ENV_FILE="$RUNTIME_DIR/service.env"
AGENT_FILE="$RUNTIME_DIR/agent-connection.env"
[[ ! -e "$ENV_FILE" && ! -e "$AGENT_FILE" ]] || {
  echo "Existing configuration found under $RUNTIME_DIR; refusing to overwrite/rotate credentials." >&2
  echo 'Edit the existing file, or deliberately choose another BUMI_RUNTIME_DIR.' >&2
  exit 2
}
read -r -s -p 'DeepSeek API key (hidden input; retained on this server only): ' DEEPSEEK_KEY
printf '\n'
[[ -n "$DEEPSEEK_KEY" ]] || { echo 'No DeepSeek key supplied' >&2; exit 2; }
SERVICE_KEY="$("$PYTHON_PATH" -c 'import secrets; print(secrets.token_urlsafe(32))')"
umask 077
mkdir -p "$RUNTIME_DIR" "$RUNTIME_DIR/data-test" "$RUNTIME_DIR/logs"
set -o noclobber
{
  printf '# Private memory service configuration. Never commit or send this entire file.\n'
  printf 'BUMI_HOST=%q\n' "${BUMI_HOST:-127.0.0.1}"
  printf 'BUMI_PORT=%q\n' "${BUMI_PORT:-8088}"
  printf 'BUMI_AGENT_ID=%q\n' "${BUMI_AGENT_ID:-bumi_robot_001}"
  printf 'BUMI_DATA_DIR=%q\n' "$RUNTIME_DIR/data-test"
  printf 'BUMI_EMBED_MODEL_PATH=%q\n' "$MODEL_DIR"
  printf 'MEM0_EMBED_DEVICE=%q\n' "${MEM0_EMBED_DEVICE:-auto}"
  printf 'BUMI_PYTHON=%q\n' "$PYTHON_PATH"
  printf 'TZ=%q\n' "${TZ:-Asia/Shanghai}"
  printf 'BUMI_SERVICE_API_KEY=%q\n' "$SERVICE_KEY"
  printf 'BUMI_MEMORY_PROVIDER=deepseek\n'
  printf 'BUMI_MEMORY_BASE_URL=https://api.deepseek.com\n'
  printf 'BUMI_MEMORY_MODEL=%q\n' "${BUMI_MEMORY_MODEL:-deepseek-flash}"
  printf 'BUMI_MEMORY_API_KEY=%q\n' "$DEEPSEEK_KEY"
  printf 'BUMI_SUMMARY_MAX_TOKENS=512\nBUMI_MEMORY_MAX_TOKENS=4096\n'
  printf 'BUMI_MEMORY_READ_TIMEOUT=0.2\nBUMI_EXPLICIT_RECALL_TIMEOUT=1.0\n'
  printf 'BUMI_MEMORY_WARMUP_TIMEOUT=120\n'
} > "$ENV_FILE"
{
  printf '# Share only this file through a private channel. No DeepSeek key is included.\n'
  printf '# Fill BUMI_SERVICE_URL with the address reachable from the Agent machine.\n'
  printf 'BUMI_SERVICE_URL=%q\n' "${BUMI_AGENT_SERVICE_URL:-}"
  printf 'BUMI_SERVICE_API_KEY=%q\n' "$SERVICE_KEY"
  printf 'BUMI_EXPECTED_AGENT_ID=%q\n' "${BUMI_AGENT_ID:-bumi_robot_001}"
} > "$AGENT_FILE"
chmod 600 "$ENV_FILE" "$AGENT_FILE"
unset DEEPSEEK_KEY SERVICE_KEY
printf 'Created private configuration: %s\n' "$ENV_FILE"
printf 'Created Agent-only connection file: %s\n' "$AGENT_FILE"
printf 'Data directory: %s (NEW TEST STORE, not your existing memories)\n' "$RUNTIME_DIR/data-test"
printf 'Default bind is loopback; for trusted LAN access explicitly set BUMI_HOST=0.0.0.0 and restrict firewall sources.\n'
printf 'No service was started, no port/firewall was changed, and no key was printed.\n'
