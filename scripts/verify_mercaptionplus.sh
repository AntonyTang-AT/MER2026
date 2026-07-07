#!/usr/bin/env bash
# 校验 MER-Caption+ CSV 与媒体（12 zip + 解压覆盖率）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
FAIL=0
STRICT_MEDIA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict-media) STRICT_MEDIA=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--strict-media]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

err() { echo "FAIL: $*"; FAIL=1; }
ok()  { echo "OK: $*"; }

echo "========== MER-Caption+ CSV =========="
CSV="${DEST}/track2_train_mercaptionplus.csv"
if [[ -f "${CSV}" ]]; then
  lines=$(wc -l < "${CSV}")
  if [[ "${lines}" -eq 31328 ]]; then
    ok "track2_train_mercaptionplus.csv lines=31328"
  else
    err "track2_train_mercaptionplus.csv lines=${lines} (expected 31328)"
  fi
else
  err "missing track2_train_mercaptionplus.csv"
fi

FILTERED="${DEST}/track2_train_mercaptionplus_filtered.csv"
if [[ -f "${FILTERED}" ]]; then
  ok "filtered csv present ($(wc -l < "${FILTERED}") lines)"
else
  echo "INFO: filtered csv not found (run scripts/filter_mercaptionplus.sh)"
fi

echo "========== MER-Caption+ ZIP (12) =========="
ZIP_DIRS=(
  "audio_7z/audio_track2_train_mercaptionplus"
  "video_7z/video_track2_train_mercaptionplus"
  "openface_7z/openface_track2_train_mercaptionplus"
)
ZIP_COUNT=0
for dir in "${ZIP_DIRS[@]}"; do
  full="${DEST}/${dir}"
  if [[ -d "${full}" ]]; then
    while IFS= read -r zip; do
      if 7z t "${zip}" >/dev/null 2>&1; then
        ok "zip ${dir}/$(basename "${zip}")"
        ZIP_COUNT=$((ZIP_COUNT + 1))
      else
        err "zip corrupt ${zip}"
      fi
    done < <(find "${full}" -maxdepth 1 -name '*.zip' | sort)
  else
    if [[ "${STRICT_MEDIA}" -eq 1 ]]; then
      err "missing dir ${dir}"
    else
      echo "INFO: missing dir ${dir} (media not downloaded)"
    fi
  fi
done
if [[ "${ZIP_COUNT}" -eq 12 ]]; then
  ok "zip count=12"
elif [[ "${ZIP_COUNT}" -gt 0 ]]; then
  err "zip count=${ZIP_COUNT} (expected 12)"
elif [[ "${STRICT_MEDIA}" -eq 1 ]]; then
  err "no MER-Caption+ media zips found"
else
  echo "INFO: MER-Caption+ media zips not downloaded yet"
fi

echo "========== MEDIA COVERAGE (optional) =========="
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python3 << PY
from pathlib import Path
from src.training.data_filter import (
    FilterConfig,
    MediaRoots,
    apply_filters,
    default_mercaptionplus_csv,
    load_mercaptionplus_table,
)

fail = False
def e(msg):
    global fail
    print(f"FAIL: {msg}")
    fail = True
def o(msg):
    print(f"OK: {msg}")

dest = Path("${DEST}")
if not default_mercaptionplus_csv(dest).is_file():
    raise SystemExit(0)

df = load_mercaptionplus_table(default_mercaptionplus_csv(dest))
roots = MediaRoots(audio=dest/"audio", video=dest/"video", face=dest/"openface_face")
filtered, summary = apply_filters(df, FilterConfig(require_media=True), media_roots=roots)
available = summary.output_count
o(f"media-ready samples {available}/{summary.input_count}")

if available == 0:
    print("INFO: no MER-Caption+ media extracted yet")
elif available < 28000:
    print(f"INFO: partial media coverage ({available} samples)")
else:
    o("media coverage looks complete")

import sys
sys.exit(1 if fail else 0)
PY
(( $? != 0 )) && FAIL=1

echo "========== SUMMARY =========="
if (( FAIL )); then
  echo "OVERALL: FAIL"
  exit 1
fi
echo "OVERALL: PASS"
exit 0
