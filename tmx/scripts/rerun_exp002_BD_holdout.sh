#!/usr/bin/env bash
# exp002 B/D 重跑 — hold-out 新模型 best epoch
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

CUDA="${1:-0}"
BEST_JSON="${ROOT}/experiments/exp001_baseline/best_epoch.json"
EPOCH="$(python3 -c "import json; print(json.load(open('${BEST_JSON}'))['epoch'])")"
CKPT_ROOT="$(python3 -c "import json; from pathlib import Path; p=json.load(open('${BEST_JSON}'))['checkpoint']; print(Path(p).parent)")"
TRACK2="${ROOT}/third_party/MERTools/MER2026/MER2026_Track2"
RUN_NAME="$(basename "${CKPT_ROOT}")"

echo "=== exp002 B/D rerun epoch=${EPOCH} run=${RUN_NAME} ==="
echo "Checkpoint root: ${CKPT_ROOT}"

bash "${ROOT}/scripts/infer_with_prompts.sh" \
  --routing-json "${ROOT}/outputs/routing/human_routing.json" \
  --prompt-variant routing \
  --dataset Human \
  --test-epoch "${EPOCH}" \
  --ckpt-root "${CKPT_ROOT}" \
  --cuda "${CUDA}"

REASON_NPZ="$(ls "${TRACK2}/output/results-mer2026ov-human-prompts/${RUN_NAME}/checkpoint_$(printf '%06d' "${EPOCH}")"*.npz 2>/dev/null | grep -v openset | head -1)"
if [[ -z "${REASON_NPZ}" || ! -f "${REASON_NPZ}" ]]; then
  echo "ERROR: reason npz not found under human-prompts/${RUN_NAME}"
  exit 1
fi
echo "REASON: ${REASON_NPZ}"

OPENSET_B="${REASON_NPZ%.npz}-openset-B.npz"
OPENSET_D="${REASON_NPZ%.npz}-openset-D.npz"

echo "=== Variant B: official ovlabel ==="
python -m src.inference.openset_extractor \
  --reason-npz "${REASON_NPZ}" \
  --store-npz "${OPENSET_B}" \
  --prompt-variant official --postprocess official \
  --cuda "${CUDA}"
bash "${ROOT}/scripts/eval_baseline.sh" "${OPENSET_B}" --split val --model-name exp002_B | tee "${ROOT}/experiments/exp002_prompt_ablation/runs/exp002_B_val_rerun.log"

echo "=== Variant D: ew_aware ovlabel ==="
python -m src.inference.openset_extractor \
  --reason-npz "${REASON_NPZ}" \
  --store-npz "${OPENSET_D}" \
  --prompt-variant ew_aware --postprocess ew \
  --cuda "${CUDA}"
bash "${ROOT}/scripts/eval_baseline.sh" "${OPENSET_D}" --split val --model-name exp002_D | tee "${ROOT}/experiments/exp002_prompt_ablation/runs/exp002_D_val_rerun.log"

python3 << PY
import json, re
from pathlib import Path
root = Path("${ROOT}")
runs = root / "experiments/exp002_prompt_ablation/runs"

def parse_metrics(log_path, model):
    text = log_path.read_text() if log_path.is_file() else ""
    m = re.search(r"EW-F1=([\d.]+)%\\s+P=([\d.]+)%\\s+R=([\d.]+)%", text)
    if not m:
        return None
    return {
        "model": model,
        "dataset": "human_ov",
        "split": "val",
        "ew_f1": float(m.group(1)) / 100,
        "precision": float(m.group(2)) / 100,
        "recall": float(m.group(3)) / 100,
        "pred_npz": Path("${OPENSET_B}" if model == "exp002_B" else "${OPENSET_D}").name,
        "reason_npz": Path("${REASON_NPZ}").as_posix().split("output/")[-1],
        "notes": f"hold-out model epoch ${EPOCH}; routing prompt; rerun 2026-07-08",
    }

for model, log in [("exp002_B", runs / "exp002_B_val_rerun.log"), ("exp002_D", runs / "exp002_D_val_rerun.log")]:
    rec = parse_metrics(log, model)
    if rec:
        (runs / f"{model}_val.json").write_text(json.dumps(rec, indent=2) + "\\n", encoding="utf-8")
        print(f"Wrote {model}_val.json EW-F1={rec['ew_f1']*100:.2f}%")
PY

echo "=== exp002 B/D rerun done ==="
