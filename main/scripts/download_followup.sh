#!/usr/bin/env bash
# 阶段 1（9 zip + 解压）完成后的后续下载：CSV、校验等
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
LOG="${ROOT}/logs/download_followup.log"
MAX_RETRIES="${MAX_RETRIES:-5}"
HF_BIN="${HF_BIN:-/root/miniconda3/bin/hf}"

mkdir -p "${ROOT}/logs"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN first" >&2
  exit 1
fi

[[ -x "${HF_BIN}" ]] || HF_BIN="$(command -v hf)"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

hf_download() {
  source /etc/network_turbo 2>/dev/null || true
  export HF_HUB_DISABLE_XET=1
  "${HF_BIN}" download "$@" --token "${HF_TOKEN}"
}

csv_ok() {
  local f="$1"
  local min_lines="$2"
  [[ -f "${DEST}/${f}" ]] || return 1
  local n
  n=$(wc -l < "${DEST}/${f}")
  (( n >= min_lines ))
}

download_csv() {
  local rel="$1"
  local min_lines="$2"
  local flog="${ROOT}/logs/download_followup_$(echo "${rel}" | tr '/.' '_').log"

  if csv_ok "${rel}" "${min_lines}"; then
    log "SKIP csv OK: ${rel} ($(wc -l < "${DEST}/${rel}") lines)"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    log "TRY ${attempt}/${MAX_RETRIES} csv: ${rel}"
    if hf_download MERChallenge/MER2026 "${rel}" \
      --repo-type dataset \
      --local-dir "${DEST}" \
      --max-workers 2 >> "${flog}" 2>&1 && csv_ok "${rel}" "${min_lines}"; then
      log "OK csv: ${rel} ($(wc -l < "${DEST}/${rel}") lines)"
      return 0
    fi
    log "FAIL csv attempt ${attempt}: ${rel}"
    ((attempt++))
    sleep $((attempt * 10))
  done
  log "GIVE UP csv: ${rel}"
  return 1
}

log "=== download_followup start ==="
df -h "${DEST}" | tee -a "${LOG}"

fail=0

# P2: MER-Caption+ 标签（阶段 5 训练用；体积小，先下好）
download_csv "track2_train_mercaptionplus.csv" 31328 || fail=1

# 可选：确保 emotion_wheel 评估资源（阶段 1 已做，此处幂等）
if [[ -x "${ROOT}/scripts/sync_emotion_wheel.sh" ]]; then
  log "=== sync emotion_wheel (idempotent) ==="
  bash "${ROOT}/scripts/sync_emotion_wheel.sh" >> "${LOG}" 2>&1 || fail=1
fi

log "=== followup CSV summary ==="
for f in track2_train_human.csv track1_track2_candidate.csv track2_train_mercaptionplus.csv subtitle_chieng.csv; do
  if [[ -f "${DEST}/${f}" ]]; then
    log "  ${f}: $(wc -l < "${DEST}/${f}") lines"
  else
    log "  MISSING: ${f}"
    fail=1
  fi
done

log "=== download_followup done (exit=${fail}) ==="
exit "${fail}"
