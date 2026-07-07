#!/usr/bin/env bash
# MER-Caption+ 12 zip 并行下载（按模态分组，避免 12 路 CAS 超时）
# 用法: PARALLEL_MODALITIES=3 bash scripts/download_mercaptionplus_parallel.sh
#       EXTRACT=1 下载完成后自动解压
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
LOG="${ROOT}/logs/download_mercaptionplus_parallel.log"
PARALLEL_MODALITIES="${PARALLEL_MODALITIES:-3}"
MAX_RETRIES="${MAX_RETRIES:-10}"
EXTRACT="${EXTRACT:-0}"
HF_BIN="${HF_BIN:-/root/miniconda3/envs/vllm3/bin/hf}"

mkdir -p "${ROOT}/logs"

source /etc/network_turbo 2>/dev/null || true
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
  fi
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN missing" >&2; exit 1; }
[[ -x "${HF_BIN}" ]] || HF_BIN="$(command -v hf)"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

zip_ok() {
  local f="$1"
  [[ -f "$f" ]] && 7z t "$f" >/dev/null 2>&1
}

hf_fetch() {
  source /etc/network_turbo 2>/dev/null || true
  export HF_HUB_DISABLE_XET=1
  export HF_HUB_ENABLE_HF_TRANSFER=0
  "${HF_BIN}" download MERChallenge/MER2026 "${1}" \
    --repo-type dataset \
    --token "${HF_TOKEN}" \
    --local-dir "${DEST}" \
    --max-workers 1
}

download_one_zip() {
  local rel="$1"
  local out="${DEST}/${rel}"
  local tag
  tag=$(echo "${rel}" | tr '/.' '_')
  local flog="${ROOT}/logs/download_mp_${tag}.log"

  if zip_ok "${out}"; then
    log "[skip] ${rel} ($(du -h "${out}" | cut -f1))"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    log "[try ${attempt}/${MAX_RETRIES}] ${rel}"
    if hf_fetch "${rel}" >> "${flog}" 2>&1 && zip_ok "${out}"; then
      log "[ok] ${rel} ($(du -h "${out}" | cut -f1))"
      return 0
    fi
    log "[fail ${attempt}] ${rel} — see ${flog}"
    ((attempt++))
    sleep $((attempt * 10))
  done
  log "[giveup] ${rel}"
  return 1
}

# 单模态内串行（断点续传），模态之间并行
download_modality() {
  local name="$1"
  shift
  local zips=("$@")
  local fail=0
  log "=== modality ${name} start (${#zips[@]} zips) ==="
  for rel in "${zips[@]}"; do
    download_one_zip "${rel}" || fail=1
  done
  log "=== modality ${name} done (fail=${fail}) ==="
  return "${fail}"
}

AUDIO_ZIPS=(
  "audio_7z/audio_track2_train_mercaptionplus/audio_split_0001.zip"
  "audio_7z/audio_track2_train_mercaptionplus/audio_split_0002.zip"
  "audio_7z/audio_track2_train_mercaptionplus/audio_split_0003.zip"
  "audio_7z/audio_track2_train_mercaptionplus/audio_split_0004.zip"
)
VIDEO_ZIPS=(
  "video_7z/video_track2_train_mercaptionplus/video_split_0001.zip"
  "video_7z/video_track2_train_mercaptionplus/video_split_0002.zip"
  "video_7z/video_track2_train_mercaptionplus/video_split_0003.zip"
  "video_7z/video_track2_train_mercaptionplus/video_split_0004.zip"
)
OPENFACE_ZIPS=(
  "openface_7z/openface_track2_train_mercaptionplus/openface_split_0004.zip"
  "openface_7z/openface_track2_train_mercaptionplus/openface_split_0001.zip"
  "openface_7z/openface_track2_train_mercaptionplus/openface_split_0002.zip"
  "openface_7z/openface_track2_train_mercaptionplus/openface_split_0003.zip"
)

log "=== MER-Caption+ parallel download start ==="
log "DEST=${DEST} PARALLEL_MODALITIES=${PARALLEL_MODALITIES}"
df -h /root/autodl-tmp | tee -a "${LOG}"

fail=0
if (( PARALLEL_MODALITIES >= 3 )); then
  download_modality audio "${AUDIO_ZIPS[@]}" &
  pid_a=$!
  download_modality video "${VIDEO_ZIPS[@]}" &
  pid_v=$!
  download_modality openface "${OPENFACE_ZIPS[@]}" &
  pid_o=$!
  wait "${pid_a}" || fail=1
  wait "${pid_v}" || fail=1
  wait "${pid_o}" || fail=1
else
  download_modality audio "${AUDIO_ZIPS[@]}" || fail=1
  download_modality video "${VIDEO_ZIPS[@]}" || fail=1
  download_modality openface "${OPENFACE_ZIPS[@]}" || fail=1
fi

ok=0
for rel in "${AUDIO_ZIPS[@]}" "${VIDEO_ZIPS[@]}" "${OPENFACE_ZIPS[@]}"; do
  zip_ok "${DEST}/${rel}" && ok=$((ok + 1))
done
log "=== zip summary: ${ok}/12 ok, fail=${fail} ==="

if [[ "${EXTRACT}" == "1" && "${ok}" -eq 12 ]]; then
  log "=== extracting track2_train_mercaptionplus ==="
  bash "${DEST}/extract_mer2026_archives.sh" "${DEST}" "${DEST}" track2_train_mercaptionplus \
    2>&1 | tee -a "${LOG}"
  bash "${ROOT}/scripts/verify_mercaptionplus.sh" --strict-media 2>&1 | tee -a "${LOG}" || fail=1
fi

log "=== finished (exit=${fail}) ==="
exit "${fail}"
