#!/usr/bin/env bash
# 本地 EW-F1 评估
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PRED="${1:-}"
if [[ -z "${PRED}" ]]; then
  echo "Usage: $0 <path/to/openset.npz> [--split val|train|all] [--analyze]"
  exit 1
fi
shift || true

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python -m src.evaluation.eval_runner --pred "${PRED}" "$@"
