#!/usr/bin/env bash
# 仅下载缺失的数据 zip 与 Qwen 分片，带重试
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
MODELS="${ROOT}/models"
LOG="${ROOT}/logs/download_missing.log"
MAX_RETRIES="${MAX_RETRIES:-8}"
HF_BIN="${HF_BIN:-/root/miniconda3/bin/hf}"

mkdir -p "${ROOT}/logs"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN first" >&2
  exit 1
fi

[[ -x "${HF_BIN}" ]] || HF_BIN="$(command -v hf)"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

hf_download() {
  # 学术加速 + 禁用 xet（避免 401）
  source /etc/network_turbo 2>/dev/null || true
  export HF_HUB_DISABLE_XET=1
  "${HF_BIN}" download "$@" --token "${HF_TOKEN}"
}

zip_ok() {
  local f="$1"
  [[ -f "$f" ]] && 7z t "$f" >/dev/null 2>&1
}

download_zip() {
  local rel="$1"
  local out="${DEST}/${rel}"
  local tag; tag=$(echo "${rel}" | tr '/.' '_')
  local flog="${ROOT}/logs/download_missing_${tag}.log"

  if zip_ok "${out}"; then
    log "SKIP zip OK: ${rel}"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    log "TRY ${attempt}/${MAX_RETRIES} zip: ${rel}"
    if hf_download MERChallenge/MER2026 "${rel}" \
      --repo-type dataset \
      --local-dir "${DEST}" \
      --max-workers 2 >> "${flog}" 2>&1 && zip_ok "${out}"; then
      log "OK zip: ${rel} ($(du -h "${out}" | cut -f1))"
      return 0
    fi
    log "FAIL zip attempt ${attempt}: ${rel}"
    ((attempt++))
    sleep $((attempt * 15))
  done
  log "GIVE UP zip: ${rel}"
  return 1
}

qwen_shard_ok() {
  local fn="$1"
  local local_f="${MODELS}/Qwen2.5-7B-Instruct/${fn}"
  [[ -f "${local_f}" ]] || return 1
  case "${fn}" in
    model-00001-of-00004.safetensors) exp=3945441440 ;;
    model-00002-of-00004.safetensors) exp=3864726352 ;;
    model-00003-of-00004.safetensors) exp=3864726424 ;;
    model-00004-of-00004.safetensors) exp=3556377672 ;;
    *) return 0 ;;
  esac
  [[ $(stat -c%s "${local_f}") -eq "${exp}" ]]
}

download_qwen_shard() {
  local fn="$1"
  local flog="${ROOT}/logs/download_missing_qwen_${fn}.log"

  if qwen_shard_ok "${fn}"; then
    log "SKIP Qwen OK: ${fn}"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    log "TRY ${attempt}/${MAX_RETRIES} Qwen: ${fn}"
    if hf_download Qwen/Qwen2.5-7B-Instruct "${fn}" \
      --local-dir "${MODELS}/Qwen2.5-7B-Instruct" \
      --max-workers 4 >> "${flog}" 2>&1 && qwen_shard_ok "${fn}"; then
      log "OK Qwen: ${fn}"
      return 0
    fi
    log "FAIL Qwen attempt ${attempt}: ${fn}"
    ((attempt++))
    sleep $((attempt * 20))
  done
  log "GIVE UP Qwen: ${fn}"
  return 1
}

MISSING_ZIPS=(
  "audio_7z/audio_track1_track2_candidate/audio_split_0002.zip"
  "video_7z/video_track2_train_human/video_split.zip"
  "video_7z/video_track1_track2_candidate/video_split_0001.zip"
  "video_7z/video_track1_track2_candidate/video_split_0002.zip"
  "openface_7z/openface_track2_train_human/openface_split.zip"
  "openface_7z/openface_track1_track2_candidate/openface_split_0001.zip"
  "openface_7z/openface_track1_track2_candidate/openface_split_0002.zip"
)

QWEN_SHARDS=(
  "model-00001-of-00004.safetensors"
  "model-00002-of-00004.safetensors"
  "model-00004-of-00004.safetensors"
)

log "=== download_missing start ==="
free -h | tee -a "${LOG}"
df -h "${DEST}" | tee -a "${LOG}"

fail=0
for fn in "${QWEN_SHARDS[@]}"; do
  download_qwen_shard "${fn}" || fail=1
done

PARALLEL_ZIPS="${PARALLEL_ZIPS:-7}"
running=0
for rel in "${MISSING_ZIPS[@]}"; do
  while (( running >= PARALLEL_ZIPS )); do
    wait -n 2>/dev/null || wait
    ((running--)) || true
  done
  download_zip "${rel}" &
  ((running++))
done
wait || true

log "=== extract archives ==="
if bash "${DEST}/extract_mer2026_archives.sh" \
  "${DEST}" "${DEST}" \
  track2_train_human track1_track2_candidate \
  >> "${LOG}" 2>&1; then
  log "extract OK"
else
  log "extract FAILED"
  fail=1
fi

log "=== verify ==="
if bash "${ROOT}/scripts/verify_downloads.sh" >> "${LOG}" 2>&1; then
  log "VERIFY PASS"
else
  log "VERIFY FAIL"
  fail=1
fi

# 若单独运行本脚本，也可手动触发：bash scripts/download_followup.sh
if [[ "${RUN_FOLLOWUP:-1}" == "1" ]] && [[ -x "${ROOT}/scripts/download_followup.sh" ]]; then
  log "=== followup downloads ==="
  if bash "${ROOT}/scripts/download_followup.sh" >> "${LOG}" 2>&1; then
    log "FOLLOWUP PASS"
  else
    log "FOLLOWUP FAIL"
    fail=1
  fi
fi

log "=== download_missing done (exit=${fail}) ==="
exit "${fail}"
