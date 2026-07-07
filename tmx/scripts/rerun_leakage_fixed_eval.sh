#!/usr/bin/env bash
# hold-out 修复后重跑：epoch 扫描 + exp002 A/C（Human val）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

CUDA="${1:-0}"
BEST_EPOCH="${2:-}"
TRACK2="${ROOT}/third_party/MERTools/MER2026/MER2026_Track2"

CKPT_ROOT="$(find "${TRACK2}/output/human_outputhybird_bestsetup_bestfusion_face_lz" -maxdepth 1 -type d -name 'human_outputhybird_*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "${CKPT_ROOT}" || ! -d "${CKPT_ROOT}" ]]; then
  echo "ERROR: no human checkpoint root under ${TRACK2}/output"
  exit 1
fi
echo "Checkpoint root: ${CKPT_ROOT}"

echo "=== Step 1: Human epoch sweep (official path, Human val) ==="
EPOCH_SWEEP_GPUS="${CUDA}" EPOCH_SWEEP_OVLABEL_GPU="${CUDA}" \
  bash "${ROOT}/scripts/sweep_epochs.sh" \
  --epochs 10-60 --skip 5 --split val \
  --gpus "${CUDA}" --ovlabel-gpu "${CUDA}" \
  --ckpt-root "${CKPT_ROOT}" \
  --no-skip-existing

if [[ -z "${BEST_EPOCH}" ]]; then
  BEST_EPOCH="$(python3 - <<PY
import json
from pathlib import Path
p = Path("${ROOT}/docs/reports/epoch_sweep_human.json")
data = json.loads(p.read_text())
best = data.get("best_epoch") or {}
print(best.get("epoch", 60))
PY
)"
fi
echo "Best epoch: ${BEST_EPOCH}"

EPOCH_TAG="$(printf 'checkpoint_%06d' "${BEST_EPOCH}")"
RESULTS_DIR="${TRACK2}/output/results-mer2026ov-human/$(basename "${CKPT_ROOT}")"
REASON_NPZ="${RESULTS_DIR}/${EPOCH_TAG}_loss_"*.npz
REASON_NPZ="$(ls ${REASON_NPZ} 2>/dev/null | grep -v openset | head -1 || true)"

if [[ -z "${REASON_NPZ}" || ! -f "${REASON_NPZ}" ]]; then
  echo "=== Step 2: official infer Human val epoch ${BEST_EPOCH} ==="
  TMX_INFERENCE_HUMAN=1 python -m src.inference.affectgpt_runner \
    --dataset Human --cuda "${CUDA}" \
    --option "inference.test_epoch=${BEST_EPOCH}" \
    --option "inference.ckpt_root=${CKPT_ROOT}"
  REASON_NPZ="$(ls "${RESULTS_DIR}/${EPOCH_TAG}"_loss_*.npz 2>/dev/null | grep -v openset | head -1)"
fi

OPENSET_A="${REASON_NPZ%.npz}-openset.npz"
OPENSET_C="${REASON_NPZ%.npz}-openset-C.npz"

echo "=== Step 3: exp002 A (official ovlabel) ==="
python -m src.inference.openset_extractor \
  --reason-npz "${REASON_NPZ}" \
  --store-npz "${OPENSET_A}" \
  --prompt-variant official --postprocess official \
  --cuda "${CUDA}"
bash "${ROOT}/scripts/eval_baseline.sh" "${OPENSET_A}" --split val --model-name exp002_A

echo "=== Step 4: exp002 C (ew_aware ovlabel) ==="
python -m src.inference.openset_extractor \
  --reason-npz "${REASON_NPZ}" \
  --store-npz "${OPENSET_C}" \
  --prompt-variant ew_aware --postprocess ew \
  --cuda "${CUDA}"
bash "${ROOT}/scripts/eval_baseline.sh" "${OPENSET_C}" --split val --model-name exp002_C

echo "Done. Update experiments/exp002_prompt_ablation/RESULTS.md with new numbers."
