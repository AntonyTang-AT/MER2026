#!/usr/bin/env bash
# exp003 混合训练实验编排
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

VARIANT=""
CUDA=""
INIT_CKPT=""
DRY_RUN=0
EXTRA_TRAIN_ARGS=()

usage() {
  cat <<EOF
Usage: $0 --variant M0|M1|M2|M3|M4|M5 [options]

Variants:
  M0  human only (exp001 reference)
  M1  mercaptionplus only
  M2  mercaptionplus filtered
  M3  mixed scratch
  M4  mixed finetune from Human best ckpt
  M5  mixed filtered finetune from Human best ckpt

Options:
  --cuda N
  --init-ckpt PATH     Required for M4/M5 (Human best checkpoint)
  --dry-run            Print planned command only
  --                   Extra args forwarded to train.py
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --cuda) CUDA="$2"; shift 2 ;;
    --init-ckpt) INIT_CKPT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    --) shift; EXTRA_TRAIN_ARGS=("$@"); break ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

[[ -z "${VARIANT}" ]] && usage

RUNS_DIR="${ROOT}/experiments/exp003_mixed_train/runs"
mkdir -p "${RUNS_DIR}"

CUDA_ARGS=()
[[ -n "${CUDA}" ]] && CUDA_ARGS=(--cuda "${CUDA}")

run_train() {
  local target="$1"
  shift
  local cmd=(bash "${ROOT}/scripts/train_affectgpt.sh" "${target}" "${CUDA_ARGS[@]}")
  cmd+=("$@")
  echo "Command: ${cmd[*]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  "${cmd[@]}"
}

case "${VARIANT}" in
  M0)
    echo "M0 uses exp001 human baseline — run: bash scripts/train_affectgpt.sh human"
    ;;
  M1)
    run_train mercaptionplus "${EXTRA_TRAIN_ARGS[@]:-}"
    ;;
  M2)
    bash "${ROOT}/scripts/filter_mercaptionplus.sh"
    run_train mercaptionplus --use-filtered "${EXTRA_TRAIN_ARGS[@]:-}"
    ;;
  M3)
    run_train mixed "${EXTRA_TRAIN_ARGS[@]:-}"
    ;;
  M4)
    [[ -z "${INIT_CKPT}" ]] && { echo "M4 requires --init-ckpt"; exit 1; }
    export TMX_FINETUNE_INIT=1
    run_train mixed -- --options "run.resume_ckpt_path=${INIT_CKPT}" "${EXTRA_TRAIN_ARGS[@]:-}"
    ;;
  M5)
    [[ -z "${INIT_CKPT}" ]] && { echo "M5 requires --init-ckpt"; exit 1; }
    export TMX_FINETUNE_INIT=1
    bash "${ROOT}/scripts/filter_mercaptionplus.sh"
    run_train mixed --use-filtered -- --options "run.resume_ckpt_path=${INIT_CKPT}" "${EXTRA_TRAIN_ARGS[@]:-}"
    ;;
  *)
    echo "Unknown variant: ${VARIANT}"
    usage
    ;;
esac

echo "Variant ${VARIANT} dispatched. Logs/results -> experiments/exp003_mixed_train/"
