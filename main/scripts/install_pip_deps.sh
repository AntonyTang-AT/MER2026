#!/usr/bin/env bash
# 补装 vllm3 pip 依赖（torch 已安装时使用）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ_FILE="${ROOT}/config/vllm3_pip_requirements.txt"
CLEAN_REQ="${ROOT}/logs/vllm3_pip_clean.txt"
LOG="${ROOT}/logs/vllm3_pip_install.log"
LOCK="${ROOT}/logs/vllm3_pip_install.lock"

mkdir -p "${ROOT}/logs"

if [[ -f "${LOCK}" ]] && kill -0 "$(cat "${LOCK}")" 2>/dev/null; then
  echo "Pip install already running (PID $(cat "${LOCK}")). tail -f ${LOG}" >&2
  exit 1
fi
echo $$ > "${LOCK}"
trap 'rm -f "${LOCK}"' EXIT

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# 过滤 conda 行、flash-attn、av（av 单独装 binary wheel）；修正 dev 版 transformers
grep -vE '^(defaults|conda-forge)$|^python=' "${REQ_FILE}" \
  | grep -vE '^flash-attn|^av==' \
  | sed 's/transformers==4.52.0.dev0/transformers==4.52.1/' > "${CLEAN_REQ}"

: > "${LOG}"
log "=== vllm3 pip deps install (retry with ffmpeg) ==="
log "Packages: $(wc -l < "${CLEAN_REQ}") + av binary (no flash-attn)"
ffmpeg -version 2>&1 | head -1 | tee -a "${LOG}" || log "WARN: ffmpeg not in PATH"
free -h | tee -a "${LOG}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true

log "=== Step 1: av binary wheel (14.2.0, replaces pinned 14.4.0) ==="
conda run -n vllm3 pip install "av==14.2.0" --only-binary=av >> "${LOG}" 2>&1 || {
  log "WARN: av binary install failed, trying 14.1.0 ..."
  conda run -n vllm3 pip install "av==14.1.0" --only-binary=av >> "${LOG}" 2>&1 || true
}

log "=== Step 2: pip install remaining requirements ==="
if conda run -n vllm3 pip install -r "${CLEAN_REQ}" \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  >> "${LOG}" 2>&1; then
  log "pip install: SUCCESS"
else
  log "WARN: pip install had errors, see log above"
fi

log "=== Step 3: post-install fixes ==="
# setuptools 82+ 移除 pkg_resources，librosa 0.8.1 与 numpy 2.x 不兼容
conda run -n vllm3 pip install "setuptools==75.8.2" "librosa==0.10.2" >> "${LOG}" 2>&1 || true

log "=== Verify ==="
conda run -n vllm3 python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "${LOG}"
for pkg in av transformers accelerate peft yaml; do
  conda run -n vllm3 python -c "import ${pkg}; v=getattr(${pkg},'__version__',None); print('${pkg}', v or 'OK')" 2>> "${LOG}" | tee -a "${LOG}" || log "${pkg} MISSING"
done
conda run -n vllm3 python -c "import vllm; print('vllm', vllm.__version__)" 2>> "${LOG}" | tee -a "${LOG}" || log "vllm MISSING (optional for some flows)"

log "=== Done ==="
