#!/usr/bin/env bash
# 两阶段创建 vllm3：conda 默认源 + pip 学术加速
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ_FILE="${ROOT}/config/vllm3_pip_requirements.txt"
LOG="${ROOT}/logs/conda_vllm3_create.log"
LOCK="${ROOT}/logs/conda_vllm3_create.lock"

mkdir -p "${ROOT}/logs"

if [[ -f "${LOCK}" ]] && kill -0 "$(cat "${LOCK}")" 2>/dev/null; then
  echo "Setup already running (PID $(cat "${LOCK}")). tail -f ${LOG}" >&2
  exit 1
fi
echo $$ > "${LOCK}"
trap 'rm -f "${LOCK}"' EXIT

log() { echo "$@" | tee -a "${LOG}"; }

: > "${LOG}"
log "=== MER2026 vllm3 setup $(date) ==="
log "=== Resource check ==="
free -h | tee -a "${LOG}"
df -h /root/autodl-tmp | tee -a "${LOG}"
nvidia-smi 2>/dev/null | tee -a "${LOG}" || log "No GPU detected"

export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
# 绕过 .condarc 中损坏的 tuna 镜像
CONDA_CHANNEL_FLAGS=(--override-channels -c defaults)

# Phase 1: conda 不用代理（避免镜像/conda 冲突）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

if conda env list | awk '{print $1}' | grep -qx "vllm3"; then
  log "Removing existing vllm3 ..."
  conda env remove -n vllm3 -y "${CONDA_CHANNEL_FLAGS[@]}" >> "${LOG}" 2>&1 || true
fi

log "=== Phase 1: conda create python=3.10 ==="
conda create -n vllm3 python=3.10 pip -y "${CONDA_CHANNEL_FLAGS[@]}" >> "${LOG}" 2>&1

log "=== Phase 2: pip install ($(wc -l < "${REQ_FILE}") packages) ==="
# pip 走官方源更快：关闭学术加速（仅 GitHub/HF 下载时开启）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true

# PyTorch CUDA 12.4 需额外 index
log "Installing PyTorch (cu124) ..."
conda run -n vllm3 pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128 >> "${LOG}" 2>&1

log "Installing remaining requirements (may take 30-60+ min) ..."
conda run -n vllm3 pip install -r "${REQ_FILE}" \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  >> "${LOG}" 2>&1 || {
    log "WARN: full pip install had errors; see log. Retrying without flash-attn if needed."
    grep -vE '^(defaults|conda-forge)$|^python=' "${REQ_FILE}" | grep -v '^flash-attn' > "${ROOT}/logs/vllm3_pip_no_flash.txt"
    conda run -n vllm3 pip install -r "${ROOT}/logs/vllm3_pip_no_flash.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      >> "${LOG}" 2>&1 || true
  }

log "=== Verify ==="
conda run -n vllm3 python --version | tee -a "${LOG}"
conda run -n vllm3 python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())" | tee -a "${LOG}" || true
conda run -n vllm3 python -c "import yaml; print('pyyaml OK')" | tee -a "${LOG}" || true
conda run -n vllm3 python -c "import transformers; print('transformers', transformers.__version__)" 2>> "${LOG}" | tee -a "${LOG}" || log "transformers not installed yet"

log "=== Finished $(date) ==="
log "Activate: conda activate vllm3"
