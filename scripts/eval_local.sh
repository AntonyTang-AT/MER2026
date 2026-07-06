#!/usr/bin/env bash
# 本地 EW-F1 评估 — 阶段 1.4 完成后启用
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PRED="${1:-}"
if [[ -z "${PRED}" ]]; then
  echo "Usage: $0 <path/to/openset.npz>"
  exit 1
fi

echo "TODO: python -m src.evaluation.eval_runner --pred ${PRED}"
