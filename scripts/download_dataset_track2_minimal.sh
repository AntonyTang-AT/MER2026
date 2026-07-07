#!/usr/bin/env bash
# Track2 最小必要数据下载（Human-OV 训练 + 候选集推理）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
LOG="${ROOT}/logs/download_dataset_track2_minimal.log"

mkdir -p "${ROOT}/logs" "${DEST}"

source /etc/network_turbo 2>/dev/null || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN first." >&2
  exit 1
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "=== Resource check ==="
free -h | tee -a "${LOG}"
df -h /root/autodl-tmp | tee -a "${LOG}"

log "=== Download CSV + Track2 Human-OV + candidate packages ==="
# Human-OV: 1532 样本，每模态 1 个 zip
# Candidate: 20000 样本，每模态 2 个 zip（提交必需）
hf download MERChallenge/MER2026 \
  --repo-type dataset \
  --token "${HF_TOKEN}" \
  --local-dir "${DEST}" \
  --include \
    "*.csv" \
    "README.md" \
    "README_AFTER_APPROVAL.md" \
    "extract_mer2026_archives.sh" \
    "audio_7z/audio_track2_train_human/**" \
    "video_7z/video_track2_train_human/**" \
    "openface_7z/openface_track2_train_human/**" \
    "audio_7z/audio_track1_track2_candidate/**" \
    "video_7z/video_track1_track2_candidate/**" \
    "openface_7z/openface_track1_track2_candidate/**" \
  2>&1 | tee -a "${LOG}"

log "=== Extract archives ==="
bash "${DEST}/extract_mer2026_archives.sh" \
  "${DEST}" "${DEST}" \
  track2_train_human track1_track2_candidate \
  2>&1 | tee -a "${LOG}"

log "=== Verify ==="
for f in track2_train_human.csv track1_track2_candidate.csv subtitle_chieng.csv; do
  [[ -f "${DEST}/${f}" ]] && log "OK csv: ${f}" || log "MISSING csv: ${f}"
done
for d in audio video openface_face; do
  if [[ -d "${DEST}/${d}" ]]; then
    log "OK dir: ${d} ($(find "${DEST}/${d}" -type f 2>/dev/null | wc -l) files)"
  else
    log "MISSING dir: ${d}"
  fi
done

log "=== Done ==="
