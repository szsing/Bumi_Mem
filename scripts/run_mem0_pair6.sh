#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
cd "$ROOT"
export HOME="$MEM0_HOME_ROOT" XDG_CACHE_HOME="$MEM0_CACHE_ROOT" PIP_CACHE_DIR="$MEM0_CACHE_ROOT/pip" HF_HOME="$MEM0_CACHE_ROOT/huggingface" HUGGINGFACE_HUB_CACHE="$MEM0_CACHE_ROOT/huggingface/hub" TORCH_HOME="$MEM0_CACHE_ROOT/torch" VLLM_CACHE_ROOT="$MEM0_CACHE_ROOT/vllm" TRITON_CACHE_DIR="$MEM0_CACHE_ROOT/triton" CONDA_PKGS_DIRS="$MEM0_CONDA_ROOT/pkgs" CONDA_ENVS_PATH="$MEM0_CONDA_ROOT/envs" TMPDIR="$MEM0_TMP_ROOT" MEM0_TELEMETRY=False TOKENIZERS_PARALLELISM=false
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$MEM0_RESULTS_ROOT/longmemeval_pair6"
exec "$MEM0_PYTHON" "$ROOT/apps/run_mem0_pair6.py" --root "$ROOT" --dataset "/home/legendre/data/ZhixingSun/dataset/longmemeval_s_cleaned.json" "$@"
