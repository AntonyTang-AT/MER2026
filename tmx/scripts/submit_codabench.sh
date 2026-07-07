#!/usr/bin/env bash
# 生成 CodaBench 提交文件
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PRED="${1:-}"
if [[ -z "${PRED}" ]]; then
  echo "Usage: $0 <path/to/openset.npz> [--out outputs/submissions/track2.csv]"
  exit 1
fi
shift || true

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python -m src.data.submission_formatter --pred "${PRED}" "$@"
echo "Output dir: ${ROOT}/outputs/submissions/"
