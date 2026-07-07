#!/usr/bin/env bash
# 下载 AffectGPT 所需核心模型（三模型并行 + hf_transfer 加速）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${ROOT}/models"
LOG="${ROOT}/logs/download_models.log"
mkdir -p "${MODELS}" "${ROOT}/logs"

check_resources() {
  echo "=== Resource check $(date) ===" | tee -a "${LOG}"
  free -h | tee -a "${LOG}"
  df -h /root/autodl-tmp | tee -a "${LOG}"
  nvidia-smi 2>/dev/null | tee -a "${LOG}" || echo "No GPU" | tee -a "${LOG}"
}

model_complete() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] || return 1
  # 有分片权重或单文件权重才算完整
  compgen -G "${dir}/model*.safetensors" >/dev/null && return 0
  compgen -G "${dir}/model*.bin" >/dev/null && return 0
  return 1
}

download_model() {
  local repo="$1"
  local dir="$2"
  local tag="${repo//\//_}"
  local mlog="${ROOT}/logs/download_model_${tag}.log"

  if model_complete "${dir}"; then
    echo "Skip complete: ${dir}" | tee -a "${LOG}"
    return 0
  fi

  echo "Downloading ${repo} -> ${dir}" | tee -a "${LOG}"
  hf download "${repo}" \
    --local-dir "${dir}" \
    --token "${HF_TOKEN:-}" \
    --max-workers 8 \
    2>&1 | tee -a "${mlog}" | tee -a "${LOG}"
}

main() {
  check_resources
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true
  pip install -q hf_transfer "huggingface_hub[cli]" 2>&1 | tee -a "${LOG}" || true

  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
  export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

  source /etc/network_turbo 2>/dev/null || true

  # 清理不完整的大模型缓存（有 config 但无权重）
  if [[ -d "${MODELS}/Qwen2.5-7B-Instruct" ]] && ! model_complete "${MODELS}/Qwen2.5-7B-Instruct"; then
    echo "Removing incomplete Qwen cache ..." | tee -a "${LOG}"
    rm -rf "${MODELS}/Qwen2.5-7B-Instruct"
  fi

  echo "=== Parallel model download $(date) ===" | tee -a "${LOG}"
  download_model "Qwen/Qwen2.5-7B-Instruct" "${MODELS}/Qwen2.5-7B-Instruct" &
  pid_qwen=$!
  download_model "openai/clip-vit-large-patch14" "${MODELS}/clip-vit-large-patch14" &
  pid_clip=$!
  download_model "TencentGameMate/chinese-hubert-large" "${MODELS}/chinese-hubert-large" &
  pid_hubert=$!

  wait "$pid_qwen" "$pid_clip" "$pid_hubert"

  echo "=== Models download finished $(date) ===" | tee -a "${LOG}"
  for d in Qwen2.5-7B-Instruct clip-vit-large-patch14 chinese-hubert-large; do
    if model_complete "${MODELS}/${d}"; then
      echo "OK: ${d}" | tee -a "${LOG}"
    else
      echo "MISSING: ${d}" | tee -a "${LOG}"
    fi
  done
}

main "$@"
