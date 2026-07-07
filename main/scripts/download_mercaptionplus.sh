#!/usr/bin/env bash
# MER-Caption+ 媒体下载 + 解压（需 HF_TOKEN；默认不自动执行，需手动调用）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
LOG="${ROOT}/logs/download_mercaptionplus.log"

mkdir -p "${ROOT}/logs" "${DEST}"

source /etc/network_turbo 2>/dev/null || true

# 禁用 HF XET/CAS 下载（部分环境证书链不兼容）
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${ROOT}/config/hf_token.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/config/hf_token.env"
  elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN first (or huggingface-cli login)." >&2
  exit 1
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "=== Resource check ==="
free -h | tee -a "${LOG}"
df -h /root/autodl-tmp | tee -a "${LOG}"

log "=== Download MER-Caption+ CSV + media zips ==="
hf download MERChallenge/MER2026 \
  --repo-type dataset \
  --token "${HF_TOKEN}" \
  --local-dir "${DEST}" \
  --include \
    "track2_train_mercaptionplus.csv" \
    "audio_7z/audio_track2_train_mercaptionplus/**" \
    "video_7z/video_track2_train_mercaptionplus/**" \
    "openface_7z/openface_track2_train_mercaptionplus/**" \
    "extract_mer2026_archives.sh" \
  2>&1 | tee -a "${LOG}"

log "=== Extract track2_train_mercaptionplus ==="
bash "${DEST}/extract_mer2026_archives.sh" \
  "${DEST}" "${DEST}" track2_train_mercaptionplus \
  2>&1 | tee -a "${LOG}"

log "=== Verify ==="
bash "${ROOT}/scripts/verify_mercaptionplus.sh" 2>&1 | tee -a "${LOG}"

log "=== Done ==="
