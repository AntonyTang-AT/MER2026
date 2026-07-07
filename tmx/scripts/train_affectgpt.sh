#!/usr/bin/env bash
# AffectGPT 训练（Human-OV 或 MER-Caption+）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

TARGET="${1:-human}"
shift || true

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

# human | mercaptionplus | mixed
python -m src.training.train_affectgpt --target "${TARGET}" "$@"
