#!/usr/bin/env bash
# 7 个缺失 zip 全并行下载（与 Qwen 等其他任务可同时进行）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
LOG="${ROOT}/logs/download_zips_parallel.log"
PARALLEL_JOBS="${PARALLEL_JOBS:-7}"
MAX_RETRIES="${MAX_RETRIES:-8}"
HF_BIN="${HF_BIN:-/root/miniconda3/bin/hf}"

mkdir -p "${ROOT}/logs"
[[ -x "${HF_BIN}" ]] || HF_BIN="$(command -v hf)"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN first" >&2
  exit 1
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

hf_download() {
  source /etc/network_turbo 2>/dev/null || true
  export HF_HUB_DISABLE_XET=1
  "${HF_BIN}" download "$@" --token "${HF_TOKEN}"
}

zip_ok() {
  local f="$1"
  [[ -f "$f" ]] && 7z t "$f" >/dev/null 2>&1
}

download_one() {
  local rel="$1"
  local out="${DEST}/${rel}"
  local tag; tag=$(echo "${rel}" | tr '/.' '_')
  local flog="${ROOT}/logs/download_zip_${tag}.log"

  if zip_ok "${out}"; then
    echo "[skip] ${rel}" >> "${LOG}"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo "[try ${attempt}] ${rel}" >> "${LOG}"
    if hf_download MERChallenge/MER2026 "${rel}" \
      --repo-type dataset \
      --local-dir "${DEST}" \
      --max-workers 2 >> "${flog}" 2>&1 && zip_ok "${out}"; then
      echo "[ok] ${rel} $(du -h "${out}" | cut -f1)" >> "${LOG}"
      return 0
    fi
    echo "[fail ${attempt}] ${rel}" >> "${LOG}"
    ((attempt++))
    sleep $((attempt * 15))
  done
  echo "[giveup] ${rel}" >> "${LOG}"
  return 1
}

export -f download_one hf_download zip_ok log
export ROOT DEST LOG HF_TOKEN HF_BIN MAX_RETRIES

ZIPS=(
  "audio_7z/audio_track1_track2_candidate/audio_split_0002.zip"
  "video_7z/video_track2_train_human/video_split.zip"
  "video_7z/video_track1_track2_candidate/video_split_0001.zip"
  "video_7z/video_track1_track2_candidate/video_split_0002.zip"
  "openface_7z/openface_track2_train_human/openface_split.zip"
  "openface_7z/openface_track1_track2_candidate/openface_split_0001.zip"
  "openface_7z/openface_track1_track2_candidate/openface_split_0002.zip"
)

log "=== parallel zip download (jobs=${PARALLEL_JOBS}) ==="
free -h | tee -a "${LOG}"

fail=0
printf '%s\n' "${ZIPS[@]}" | xargs -P "${PARALLEL_JOBS}" -I {} bash -c 'download_one "$@"' _ {}

# 统计
ok=0; miss=0
for rel in "${ZIPS[@]}"; do
  zip_ok "${DEST}/${rel}" && ok=$((ok+1)) || miss=$((miss+1))
done
log "=== done: ok=${ok}/7 missing=${miss} ==="
exit $(( miss > 0 ? 1 : 0 ))
