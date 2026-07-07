#!/usr/bin/env bash
# Baseline 全流程：sync -> infer -> ovlabel -> eval（需已训练 checkpoint）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

echo "=== Step 1: sync MERTools config ==="
bash "${ROOT}/scripts/sync_mertools_config.sh"

echo "=== Step 2: AffectGPT inference ==="
bash "${ROOT}/scripts/infer_affectgpt.sh" "$@"

echo "=== Step 3: ovlabel extraction (all pending npz) ==="
bash "${ROOT}/scripts/run_ovlabel.sh" --all

echo "=== Step 4: find latest openset npz and eval ==="
TRACK2="${ROOT}/third_party/MERTools/MER2026/MER2026_Track2"
LATEST="$(find "${TRACK2}/output" -name '*-openset.npz' 2>/dev/null | sort | tail -1 || true)"
if [[ -z "${LATEST}" ]]; then
  echo "WARN: no -openset.npz found under ${TRACK2}/output; run eval manually"
  exit 0
fi
echo "Evaluating: ${LATEST}"
bash "${ROOT}/scripts/eval_baseline.sh" "${LATEST}" --split val
