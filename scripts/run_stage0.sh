#!/usr/bin/env bash
set -euo pipefail
set +x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE="$(dirname "$ROOT")"
PY="${STAGE0_PYTHON:-$WORKSPACE/mem0-runtime/conda/envs/mem0/bin/python}"
RUNS="$WORKSPACE/mem0-runtime/stage0/runs"
mkdir -p -- "$RUNS"
RUN="${STAGE0_RUN_DIR:-$(mktemp -d "$RUNS/manual-$(date +%Y%m%dT%H%M%S)-XXXXXX")}"
export STAGE0_RUN_DIR="$RUN"
cd -- "$ROOT"
ACTION="${1:-all}"
printf 'Stage-0 run directory: %s\n' "$RUN"
if [[ "$ACTION" == all ]]; then
  for command in doctor validate baseline contract-tests storage-probe; do
    "$PY" tools/stage0/stage0.py --run-dir "$RUN" "$command"
  done
elif [[ "$ACTION" == probe-cpu || "$ACTION" == probe-cuda ]]; then
  DEVICE="${ACTION#probe-}"
  if [[ "$DEVICE" == cuda ]]; then
    PY="${STAGE0_GPU_PYTHON:-$WORKSPACE/mem0-runtime/conda/envs/vllm/bin/python}"
  fi
  "$PY" tools/stage0/embedding_probe.py --device "$DEVICE" \
    --model-path "$WORKSPACE/models/Qwen3-Embedding-0.6B" \
    --output "$RUN/reports/embedding_${DEVICE}.json"
else
  case "$ACTION" in doctor|validate|baseline|contract-tests|storage-probe) ;;
    *) echo 'Use all, doctor, validate, baseline, contract-tests, storage-probe, probe-cpu or probe-cuda.' >&2; exit 2;;
  esac
  "$PY" tools/stage0/stage0.py --run-dir "$RUN" "$ACTION"
fi
