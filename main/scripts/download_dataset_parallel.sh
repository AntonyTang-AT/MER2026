#!/usr/bin/env bash
# Track2 最小必要数据 — 9 zip 并行下载（bash job pool + hf --max-workers）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
LOG="${ROOT}/logs/download_dataset_parallel.log"
PARALLEL_JOBS="${PARALLEL_JOBS:-9}"
WORKERS_PER_FILE="${WORKERS_PER_FILE:-4}"

mkdir -p "${ROOT}/logs" "${DEST}"

source /etc/network_turbo 2>/dev/null || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN first: export HF_TOKEN=hf_xxx" >&2
  exit 1
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

download_one() {
  local file="$1"
  local flog="${ROOT}/logs/download_zip_$(echo "${file}" | tr '/.' '_').log"
  if [[ -f "${DEST}/${file}" ]]; then
    local sz
    sz=$(stat -c%s "${DEST}/${file}" 2>/dev/null || echo 0)
    if [[ "${sz}" -gt 1000000 ]]; then
      echo "[skip] ${file} (${sz} bytes)" >> "${LOG}"
      return 0
    fi
  fi
  echo "[start] ${file}" >> "${LOG}"
  hf download MERChallenge/MER2026 "${file}" \
    --repo-type dataset \
    --token "${HF_TOKEN}" \
    --local-dir "${DEST}" \
    --max-workers "${WORKERS_PER_FILE}" \
    >> "${flog}" 2>&1
  echo "[done] ${file}" >> "${LOG}"
}

export -f download_one
export ROOT DEST LOG HF_TOKEN WORKERS_PER_FILE

ZIPS=(
  "audio_7z/audio_track2_train_human/audio_split.zip"
  "audio_7z/audio_track1_track2_candidate/audio_split_0001.zip"
  "audio_7z/audio_track1_track2_candidate/audio_split_0002.zip"
  "video_7z/video_track2_train_human/video_split.zip"
  "video_7z/video_track1_track2_candidate/video_split_0001.zip"
  "video_7z/video_track1_track2_candidate/video_split_0002.zip"
  "openface_7z/openface_track2_train_human/openface_split.zip"
  "openface_7z/openface_track1_track2_candidate/openface_split_0001.zip"
  "openface_7z/openface_track1_track2_candidate/openface_split_0002.zip"
)

log "=== MER2026 Track2 parallel zip download (jobs=${PARALLEL_JOBS}, workers/file=${WORKERS_PER_FILE}) ==="
log "Target: ${DEST}"
free -h | tee -a "${LOG}"
df -h "${DEST}" | tee -a "${LOG}"

# CSV 等小文件（秒级）
for f in track2_train_human.csv track1_track2_candidate.csv subtitle_chieng.csv \
         README.md README_AFTER_APPROVAL.md extract_mer2026_archives.sh; do
  if [[ ! -f "${DEST}/${f}" ]]; then
    log ">>> ${f}"
    hf download MERChallenge/MER2026 "${f}" \
      --repo-type dataset --token "${HF_TOKEN}" \
      --local-dir "${DEST}" >> "${LOG}" 2>&1
  else
    log "skip csv/meta: ${f}"
  fi
done

log "=== Parallel download ${#ZIPS[@]} zip files ==="
printf '%s\n' "${ZIPS[@]}" | xargs -P "${PARALLEL_JOBS}" -I {} bash -c 'download_one "$@"' _ {}

log "=== Extract archives ==="
bash "${DEST}/extract_mer2026_archives.sh" \
  "${DEST}" "${DEST}" \
  track2_train_human track1_track2_candidate \
  2>&1 | tee -a "${LOG}"

log "=== Verify ==="
for f in track2_train_human.csv track1_track2_candidate.csv subtitle_chieng.csv; do
  [[ -f "${DEST}/${f}" ]] && log "OK csv: ${f} ($(wc -l < "${DEST}/${f}") lines)" || log "MISSING csv: ${f}"
done
for d in audio video openface_face; do
  if [[ -d "${DEST}/${d}" ]]; then
    log "OK dir: ${d} ($(find "${DEST}/${d}" -type f 2>/dev/null | wc -l) files)"
  else
    log "MISSING dir: ${d}"
  fi
done
log "=== Done ==="
