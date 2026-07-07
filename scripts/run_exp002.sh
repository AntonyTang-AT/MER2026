#!/usr/bin/env bash
# exp002 Prompt 消融编排
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

VARIANT=""
SPLIT="val"
CUDA=""
ROUTING_JSON="${ROOT}/outputs/routing/human_routing.json"
EVAL_ONLY=0
PRED_NPZ=""

usage() {
  echo "Usage: $0 --variant A|B|C|D [--split val|train|all] [--cuda N] [--eval-only PRED_NPZ]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --cuda) CUDA="$2"; shift 2 ;;
    --routing-json) ROUTING_JSON="$2"; shift 2 ;;
    --eval-only) EVAL_ONLY=1; PRED_NPZ="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

RUNS_DIR="${ROOT}/experiments/exp002_prompt_ablation/runs"
mkdir -p "${RUNS_DIR}"

run_eval() {
  local pred="$1"
  local tag="$2"
  local out="${RUNS_DIR}/exp002_${tag}_${SPLIT}.json"
  local cuda_args=()
  [[ -n "${CUDA}" ]] && cuda_args=(--cuda "${CUDA}")
  bash "${ROOT}/scripts/eval_baseline.sh" "${pred}" --split "${SPLIT}" --model-name "exp002_${tag}" \
    | tee "${out}.log"
  echo "Eval log: ${out}.log"
}

if [[ "${EVAL_ONLY}" -eq 1 ]]; then
  [[ -z "${PRED_NPZ}" || -z "${VARIANT}" ]] && usage
  run_eval "${PRED_NPZ}" "${VARIANT}"
  exit 0
fi

[[ -z "${VARIANT}" ]] && usage

CUDA_ARGS=()
[[ -n "${CUDA}" ]] && CUDA_ARGS=(--cuda "${CUDA}")

BEST_JSON="${ROOT}/experiments/exp001_baseline/best_epoch.json"
BEST_EPOCH=""
if [[ -f "${BEST_JSON}" ]]; then
  BEST_EPOCH="$(python3 - <<PY
import json
from pathlib import Path
print(json.loads(Path("${BEST_JSON}").read_text()).get("epoch", ""))
PY
)"
fi
HUMAN_VAL_ARGS=()
if [[ "${SPLIT}" == "val" ]]; then
  HUMAN_VAL_ARGS=(--dataset Human)
fi
EPOCH_ARGS=()
if [[ -n "${BEST_EPOCH}" ]]; then
  EPOCH_ARGS=(--test-epoch "${BEST_EPOCH}")
fi

case "${VARIANT}" in
  A)
    bash "${ROOT}/scripts/infer_baseline.sh" "${CUDA_ARGS[@]:-}"
    ;;
  B)
    bash "${ROOT}/scripts/infer_with_prompts.sh" \
      --routing-json "${ROUTING_JSON}" \
      --prompt-variant routing \
      "${HUMAN_VAL_ARGS[@]}" \
      "${EPOCH_ARGS[@]}" \
      "${CUDA_ARGS[@]:-}"
    bash "${ROOT}/scripts/run_ovlabel.sh" --all "${CUDA_ARGS[@]:-}"
    ;;
  C)
    echo "Variant C: run ovlabel on existing exp001 reason npz with ew_aware prompt"
    bash "${ROOT}/scripts/run_ovlabel.sh" --all \
      --prompt-variant ew_aware \
      --postprocess ew \
      "${CUDA_ARGS[@]:-}"
    ;;
  D)
    bash "${ROOT}/scripts/infer_with_prompts.sh" \
      --routing-json "${ROUTING_JSON}" \
      --prompt-variant routing \
      "${HUMAN_VAL_ARGS[@]}" \
      "${EPOCH_ARGS[@]}" \
      "${CUDA_ARGS[@]:-}"
    bash "${ROOT}/scripts/run_ovlabel.sh" --all \
      --prompt-variant ew_aware \
      --postprocess ew \
      "${CUDA_ARGS[@]:-}"
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    usage
    ;;
esac

TRACK2="${ROOT}/third_party/MERTools/MER2026/MER2026_Track2"
LATEST="$(find "${TRACK2}/output" -name '*-openset.npz' 2>/dev/null | sort | tail -1 || true)"
if [[ -n "${LATEST}" ]]; then
  run_eval "${LATEST}" "${VARIANT}"
else
  echo "WARN: no openset npz found; eval manually after ovlabel"
fi
