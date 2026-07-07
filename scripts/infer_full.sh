#!/usr/bin/env bash
# 完整推理流水线：sync → Stage A→B→C → submission
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

echo "=== Step 1: sync MERTools config ==="
bash "${ROOT}/scripts/sync_mertools_config.sh"

echo "=== Step 2: full pipeline (submit mode) ==="
python -m src.inference.pipeline --mode submit --profile "$@"
