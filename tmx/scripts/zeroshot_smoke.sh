#!/usr/bin/env bash
# Zero-shot MLLM 抽测（可选，需 SALMONN/Chat-UniVi 权重）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACK2="${ROOT}/third_party/MERTools/MER2026/MER2026_Track2"

echo "Zero-shot baseline (SALMONN / Chat-UniVi) requires extra model weights."
echo "See third_party/MERTools/MER2026/MER2026_Track2/README.md"
echo ""
echo "If models are installed under ${TRACK2}/models/, run manually, e.g.:"
echo "  cd ${TRACK2}/SALMONN && CUDA_VISIBLE_DEVICES=0 python main-audio.py --subtitle_flag=subtitle --dataset=MER2026OV"
echo "  cd ${TRACK2}/Chat-UniVi && CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag=subtitle --dataset=MER2026OV"
echo ""
echo "Then: python ovlabel_extraction.py && bash ${ROOT}/scripts/eval_baseline.sh <openset.npz>"
exit 0
