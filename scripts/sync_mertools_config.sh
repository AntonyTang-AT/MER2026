#!/usr/bin/env bash
# 同步 MERTools config.py 路径 + models 软链接
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python -m src.training.train_affectgpt --sync-only "$@"
