#!/usr/bin/env bash
# 完整推理流水线（Stage A→B→C）— 阶段 6 完成后启用
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "TODO: implement src/inference/pipeline.py CLI"
echo "Project root: ${ROOT}"
# python -m src.inference.pipeline --dataset MER2026OV
