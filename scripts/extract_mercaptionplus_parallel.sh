#!/usr/bin/env bash
# MER-Caption+ 并行解压 — 优先级 openface > audio > video，同模态多 zip 并行
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
JOBS="${EXTRACT_JOBS:-3}"
PACKAGE="track2_train_mercaptionplus"

usage() {
  echo "Usage: EXTRACT_JOBS=3 $0"
  echo "  openface/audio/video 按优先级；每模态内最多 JOBS 个 7z 并行"
}

require_cmd() { command -v "$1" >/dev/null || { echo "need $1"; exit 1; }; }

extract_one() {
  local archive="$1"
  local target="$2"
  if [[ ! -f "$archive" ]]; then
    echo "SKIP missing: $archive"
    return 0
  fi
  echo "[$(date '+%H:%M:%S')] START $(basename "$archive")"
  7z x -y -aoa "$archive" "-o$target" >/dev/null
  echo "[$(date '+%H:%M:%S')] DONE  $(basename "$archive")"
}

extract_modality_parallel() {
  local medium="$1"
  local dir="${DEST}/${medium}_7z/${medium}_${PACKAGE}"
  [[ -d "$dir" ]] || { echo "WARN: no dir $dir"; return 0; }
  local -a archives=()
  while IFS= read -r z; do archives+=("$z"); done < <(find "$dir" -maxdepth 1 -name "${medium}_split*.zip" | sort)
  if ((${#archives[@]} == 0)); then
    echo "WARN: no zips in $dir"
    return 0
  fi
  echo "=== ${medium}: ${#archives[@]} zips, parallel=${JOBS} ==="
  local running=0
  for archive in "${archives[@]}"; do
    extract_one "$archive" "$DEST" &
    running=$((running + 1))
    if (( running >= JOBS )); then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

main() {
  require_cmd 7z
  mkdir -p "$DEST"
  echo "DEST=$DEST JOBS=$JOBS PACKAGE=$PACKAGE"
  df -h /root/autodl-tmp | tail -1
  # face_lz 训练最急需 openface，其次 audio，video 可最后
  extract_modality_parallel openface
  extract_modality_parallel audio
  extract_modality_parallel video
  echo "All done -> $DEST"
}

main "$@"
