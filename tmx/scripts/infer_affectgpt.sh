#!/usr/bin/env bash
# AffectGPT 推理（官方 inference_hybird.py）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python -m src.inference.affectgpt_runner "$@"
