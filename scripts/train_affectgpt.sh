#!/usr/bin/env bash
# AffectGPT 训练（需先配置 data/models 路径）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACK2="${ROOT}/third_party/MERTools/MER2026/MER2026_Track2"
CFG="${1:-train_configs/human_outputhybird_bestsetup_bestfusion_face_lz.yaml}"

cd "${TRACK2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u train.py --cfg-path="${CFG}"
