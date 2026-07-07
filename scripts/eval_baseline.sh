#!/usr/bin/env bash
# Baseline 评估 + exp001 报告
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PRED="${1:-}"
if [[ -z "${PRED}" ]]; then
  echo "Usage: $0 <path/to/openset.npz> [--split val|train|all] [--model-name NAME]"
  exit 1
fi
shift || true

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python -m src.evaluation.baseline_report --pred "${PRED}" "$@"
