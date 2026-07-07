#!/usr/bin/env bash
# Track2 最小必要数据 — 串行下载（避免 hf 并行导致 OOM Kill）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
LOG="${ROOT}/logs/download_dataset_sequential.log"

mkdir -p "${ROOT}/logs" "${DEST}"

source /etc/network_turbo 2>/dev/null || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN first: export HF_TOKEN=hf_xxx" >&2
  exit 1
fi

export HF_HUB_DOWNLOAD_CONCURRENCY=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

download_one() {
  local pattern="$1"
  log ">>> Download: ${pattern}"
  hf download MERChallenge/MER2026 \
    --repo-type dataset \
    --token "${HF_TOKEN}" \
    --local-dir "${DEST}" \
    "${pattern}" \
    2>&1 | tee -a "${LOG}"
}

log "=== MER2026 Track2 minimal sequential download ==="
log "Target: ${DEST}"
free -h | tee -a "${LOG}"
df -h "${DEST}" | tee -a "${LOG}"

# CSV + 工具脚本
for f in \
  "track2_train_human.csv" \
  "track1_track2_candidate.csv" \
  "subtitle_chieng.csv" \
  "README.md" \
  "README_AFTER_APPROVAL.md" \
  "extract_mer2026_archives.sh"
do
  download_one "${f}"
done

# 9 个 zip（每模态串行）
for f in \
  "audio_7z/audio_track2_train_human/audio_split.zip" \
  "audio_7z/audio_track1_track2_candidate/audio_split_0001.zip" \
  "audio_7z/audio_track1_track2_candidate/audio_split_0002.zip" \
  "video_7z/video_track2_train_human/video_split.zip" \
  "video_7z/video_track1_track2_candidate/video_split_0001.zip" \
  "video_7z/video_track1_track2_candidate/video_split_0002.zip" \
  "openface_7z/openface_track2_train_human/openface_split.zip" \
  "openface_7z/openface_track1_track2_candidate/openface_split_0001.zip" \
  "openface_7z/openface_track1_track2_candidate/openface_split_0002.zip"
do
  download_one "${f}"
done

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
