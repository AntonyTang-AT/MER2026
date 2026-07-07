#!/usr/bin/env bash
# 后台启动 Human-OV 全量训练（60 epoch × 500 iter）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS="${PROJECT_ROOT:-${ROOT}/main}"
LOG="${ROOT}/logs/train_human_full.log"
PIDFILE="${ROOT}/logs/train_human_full.pid"
LOCK="${ROOT}/logs/train_human_full.lock"
CFG="train_configs/human_outputhybird_bestsetup_bestfusion_face_lz.yaml"
MERTOOLS="${ROOT}/third_party/MERTools/MER2026/MER2026_Track2"

mkdir -p "${ROOT}/logs"

if [[ -f "${LOCK}" ]]; then
  old_pid="$(cat "${LOCK}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "Training already running (pid ${old_pid})"
    echo "Log: ${LOG}"
    echo "Monitor: python ${ROOT}/scripts/watch_train_progress.py"
    exit 0
  fi
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

export PYTHONPATH="${WS}:${PYTHONPATH:-}"

: > "${LOG}"
echo "[$(date '+%F %T')] === start Human-OV full training ===" | tee -a "${LOG}"
echo "workspace: ${WS}" | tee -a "${LOG}"
echo "config: ${CFG} (max_epoch=60, iters_per_epoch=500, batch_size_train=3)" | tee -a "${LOG}"
free -h | tee -a "${LOG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "${LOG}" || true

nohup bash -c "
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate vllm3
  export PYTHONPATH='${WS}:'\${PYTHONPATH:-}
  cd '${MERTOOLS}'
  exec python -u -m src.training.mertools_entry train.py --cfg-path='${CFG}'
" >> "${LOG}" 2>&1 &

train_pid=$!
echo "${train_pid}" > "${PIDFILE}"
echo "${train_pid}" > "${LOCK}"

echo "Started training pid=${train_pid}"
echo "Log: ${LOG}"
echo ""
echo "实时进度（另开终端运行）:"
echo "  python ${ROOT}/scripts/watch_train_progress.py"
echo "或:"
echo "  bash ${ROOT}/scripts/watch_train_progress.sh"
