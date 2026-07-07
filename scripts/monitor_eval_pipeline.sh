#!/usr/bin/env bash
# 实时监控 eval pipeline / epoch sweep 进度
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${MONITOR_INTERVAL:-30}"

while true; do
  clear
  echo "========== $(date '+%F %T') MER2026 Pipeline Monitor =========="
  echo

  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
  echo

  echo "--- Processes ---"
  ps aux | grep -E 'epoch_sweep|mertools_entry|openset_extractor|run_eval_pipeline|download_mercaptionplus|hf download' \
    | grep -v grep | awk '{print $11, $12, $13, $14}' | head -8 || echo "(none)"
  echo

  echo "--- Epoch sweep progress ---"
  eval_dir="${ROOT}/logs/epoch_sweep/human"
  if [[ -d "${eval_dir}" ]]; then
    evaluated=$(grep -l '"status": "evaluated"' "${eval_dir}"/*.json 2>/dev/null | wc -l)
    infer_done=$(grep -l '"status": "infer_done"' "${eval_dir}"/*.json 2>/dev/null | wc -l)
    failed=$(grep -lE '"status": "(infer_failed|ovlabel_failed)"' "${eval_dir}"/*.json 2>/dev/null | wc -l)
    echo "evaluated=${evaluated}/11 infer_done=${infer_done} failed=${failed}"
    ls -1 "${eval_dir}"/epoch_*_eval*.json 2>/dev/null | tail -3 || true
  else
    echo "(no progress dir yet)"
  fi
  echo

  echo "--- Openset npz (human val) ---"
  find "${ROOT}/third_party/MERTools" -path '*results-mer2026ov-human*' -name '*-openset.npz' 2>/dev/null | wc -l | xargs echo "count:"
  echo

  echo "--- MER-Caption+ download ---"
  tail -3 "${ROOT}/logs/download_mercaptionplus.log" 2>/dev/null || echo "(no download log)"
  echo

  echo "--- Pipeline log tail ---"
  latest_pipe=$(ls -t "${ROOT}/logs/eval_pipeline"/pipeline_*.log 2>/dev/null | head -1)
  if [[ -n "${latest_pipe}" ]]; then
    tail -5 "${latest_pipe}"
  else
    latest_sweep=$(ls -t "${ROOT}/logs/epoch_sweep"/run_*.log 2>/dev/null | head -1)
    [[ -n "${latest_sweep}" ]] && tail -5 "${latest_sweep}" || echo "(no log)"
  fi

  sleep "${INTERVAL}"
done
