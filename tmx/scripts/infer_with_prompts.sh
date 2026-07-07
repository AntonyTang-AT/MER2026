#!/usr/bin/env bash
# 带 routing-aware Prompt 的 AffectGPT 推理
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python -m src.inference.infer_with_prompts "$@"
