#!/usr/bin/env bash
# Human / mixed 训练 checkpoint epoch 扫描 + val EW-F1（默认 5 卡并行）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

GPUS="${EPOCH_SWEEP_GPUS:-0,1,2,3,4}"
OVLABEL_GPU="${EPOCH_SWEEP_OVLABEL_GPU:-0,1,2,3,4}"
python -u -m src.training.epoch_sweep --gpus "${GPUS}" --ovlabel-gpu "${OVLABEL_GPU}" "$@"
