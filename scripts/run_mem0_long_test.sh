#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
PYTHON="$MEM0_PYTHON"
DEVICE="${MEM0_EMBED_DEVICE:-auto}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/src/mem0-new${PYTHONPATH:+:$PYTHONPATH}"
export MEM0_REPRO_ROOT="$ROOT"
export HOME="$MEM0_HOME_ROOT"
export XDG_CACHE_HOME="$MEM0_CACHE_ROOT"
export PIP_CACHE_DIR="$MEM0_CACHE_ROOT/pip"
export HF_HOME="$MEM0_CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$MEM0_CACHE_ROOT/huggingface/hub"
export TORCH_HOME="$MEM0_CACHE_ROOT/torch"
export VLLM_CACHE_ROOT="$MEM0_CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$MEM0_CACHE_ROOT/triton"
export UV_CACHE_DIR="$MEM0_CACHE_ROOT/uv"
export CONDA_PKGS_DIRS="$MEM0_CONDA_ROOT/pkgs"
export CONDA_ENVS_PATH="$MEM0_CONDA_ROOT/envs"
export TMPDIR="$MEM0_TMP_ROOT"
export MEM0_TELEMETRY=False
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$MEM0_DATA_ROOT/new_long_tests"
if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 MemNew 环境：$PYTHON" >&2
  exit 1
fi
if [[ " ${*:-} " != *" --dry-run "* ]] && ! curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null; then
  echo "本地 vLLM 未就绪。请先运行：scripts/start_vllm_qwen.sh" >&2
  exit 1
fi
exec "$PYTHON" "$ROOT/apps/mem0_long_test.py" \
  --root "$ROOT" \
  --case "$ROOT/tests/mem0_long_case.json" \
  --device "$DEVICE" \
  --data-subdir new_long_tests \
  --collection-name mem0_long_test \
  --top-k 5 \
  --search-threshold 0.1 \
  --memory-max-tokens 512 \
  "$@"
