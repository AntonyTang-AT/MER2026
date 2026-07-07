#!/usr/bin/env bash
# reason.npz -> openset.npz（官方 ovlabel_extraction）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python -m src.inference.openset_extractor "$@"
