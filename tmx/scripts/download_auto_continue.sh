#!/usr/bin/env bash
# 等待当前 hf 下载结束 → 补齐 zip/解压/验证 → 自动执行后续下载（followup）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/logs/download_auto_continue.log"
LOCK="${ROOT}/logs/download_auto_continue.lock"
POLL_SEC="${POLL_SEC:-60}"

mkdir -p "${ROOT}/logs"

if [[ -f "${LOCK}" ]]; then
  old_pid=$(cat "${LOCK}" 2>/dev/null || true)
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "download_auto_continue already running (pid ${old_pid})" | tee -a "${LOG}"
    exit 0
  fi
fi
echo $$ > "${LOCK}"
trap 'rm -f "${LOCK}"' EXIT

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

wait_for_idle() {
  log "=== waiting for in-flight MER2026 downloads ==="
  while true; do
    local n wrappers
    n=$(pgrep -fc 'hf download MERChallenge/MER2026' 2>/dev/null) || n=0
    wrappers=$(pgrep -fc 'download_missing\.sh|download_zips_parallel\.sh' 2>/dev/null) || wrappers=0

    if (( n == 0 && wrappers == 0 )); then
      log "no active hf/wrapper download processes"
      break
    fi
    log "still running: hf=${n} wrappers=${wrappers} (poll ${POLL_SEC}s)"
    sleep "${POLL_SEC}"
  done
}

zip_all_ok() {
  local zips=(
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
  local rel f
  for rel in "${zips[@]}"; do
    f="${ROOT}/data/mer2026-dataset/${rel}"
    if [[ ! -f "${f}" ]] || ! 7z t "${f}" >/dev/null 2>&1; then
      return 1
    fi
  done
  return 0
}

log "=== download_auto_continue start (pid $$) ==="
df -h "${ROOT}/data" | tee -a "${LOG}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${ROOT}/config/hf_token.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/config/hf_token.env"
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  log "ERROR: HF_TOKEN not set; export HF_TOKEN or create config/hf_token.env"
  exit 1
fi

wait_for_idle

fail=0

log "=== phase 1: download_missing (zip + extract + verify, idempotent) ==="
if bash "${ROOT}/scripts/download_missing.sh" >> "${LOG}" 2>&1; then
  log "download_missing PASS"
else
  log "download_missing FAIL (will still try followup if zips OK)"
  fail=1
fi

if zip_all_ok; then
  log "all 9 zips verified OK"
else
  log "WARN: not all 9 zips OK after download_missing"
  fail=1
fi

log "=== phase 2: download_followup (mercaptionplus csv, etc.) ==="
if bash "${ROOT}/scripts/download_followup.sh" >> "${LOG}" 2>&1; then
  log "download_followup PASS"
else
  log "download_followup FAIL"
  fail=1
fi

log "=== final verify ==="
if bash "${ROOT}/scripts/verify_downloads.sh" >> "${LOG}" 2>&1; then
  log "FINAL VERIFY PASS"
else
  log "FINAL VERIFY FAIL (mercaptionplus csv not in verify script — check logs)"
  # verify_downloads may fail only on zips; don't force fail if followup ok
fi

log "=== download_auto_continue done (exit=${fail}) ==="
exit "${fail}"
