#!/usr/bin/env bash
# Human / mixed 训练 checkpoint epoch 扫描 + val EW-F1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python -m src.training.epoch_sweep "$@"
