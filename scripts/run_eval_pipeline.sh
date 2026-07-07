#!/usr/bin/env bash
# Epoch 扫描 → exp001 基线报告 → exp002 Prompt 消融（不含 20k 提交推理）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
LOG_DIR="${ROOT}/logs/eval_pipeline"
mkdir -p "${LOG_DIR}"
PIPE_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

log() { echo "[$(date '+%F %T')] $*" | tee -a "${PIPE_LOG}"; }

log "=== MER2026 eval pipeline start ==="
log "log: ${PIPE_LOG}"

# --- Step 1: Epoch sweep ---
log "=== Step 1/4: epoch sweep (5-GPU infer + 5-GPU ovlabel) ==="
if ! bash "${ROOT}/scripts/sweep_epochs.sh" --train-run human --epochs 10-60 --skip 5 2>&1 | tee -a "${PIPE_LOG}"; then
  log "ERROR: epoch sweep failed — see ${PIPE_LOG}"
  exit 1
fi

BEST_JSON="${ROOT}/experiments/exp001_baseline/best_epoch.json"
if [[ ! -f "${BEST_JSON}" ]]; then
  log "ERROR: best_epoch.json not found after sweep"
  exit 1
fi

BEST_OPENSET="$(python3 - <<PY
import json
from pathlib import Path
data = json.loads(Path("${BEST_JSON}").read_text())
print(data.get("openset_npz") or "")
PY
)"
if [[ -z "${BEST_OPENSET}" || ! -f "${BEST_OPENSET}" ]]; then
  log "ERROR: best openset npz missing: ${BEST_OPENSET}"
  exit 1
fi

# --- Step 2: exp001 baseline report ---
log "=== Step 2/4: exp001 baseline report (best epoch) ==="
log "best openset: ${BEST_OPENSET}"
if ! bash "${ROOT}/scripts/eval_baseline.sh" "${BEST_OPENSET}" --split val \
  --model-name "exp001_best_epoch" 2>&1 | tee -a "${PIPE_LOG}"; then
  log "ERROR: exp001 eval failed"
  exit 1
fi

# --- Step 3: exp002 variant C (ew openset, no re-infer) ---
log "=== Step 3/4: exp002 variant C (ew_aware ovlabel) ==="
if ! bash "${ROOT}/scripts/run_exp002.sh" --variant C --split val --cuda 0 2>&1 | tee -a "${PIPE_LOG}"; then
  log "WARN: exp002 variant C failed (continuing)"
fi

# --- Step 4: exp002 variant D (routing + ew, full prompt stack) ---
log "=== Step 4/4: exp002 variant D (routing + ew_aware) ==="
if ! bash "${ROOT}/scripts/run_exp002.sh" --variant D --split val --cuda 0 2>&1 | tee -a "${PIPE_LOG}"; then
  log "WARN: exp002 variant D failed"
fi

log "=== Pipeline complete ==="
log "exp001: ${ROOT}/experiments/exp001_baseline/"
log "exp002: ${ROOT}/experiments/exp002_prompt_ablation/runs/"
log "epoch sweep: ${ROOT}/docs/reports/epoch_sweep_human.md"
