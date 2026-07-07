#!/usr/bin/env bash
# 解压 Human-OV 训练所需媒体（track2_train_human 三包）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
EXTRACT="${DEST}/extract_mer2026_archives.sh"

if [[ ! -f "${EXTRACT}" ]]; then
  echo "ERROR: missing ${EXTRACT}"
  exit 1
fi

sed -i 's/\r$//' "${EXTRACT}" 2>/dev/null || true

echo "Extracting track2_train_human -> ${DEST}"
bash "${EXTRACT}" "${DEST}" "${DEST}" track2_train_human

echo "========== verify human media =========="
for d in audio video openface_face; do
  if [[ -d "${DEST}/${d}" ]]; then
    n=$(find "${DEST}/${d}" -type f | wc -l)
    echo "OK: ${d}/ ${n} files"
  else
    echo "FAIL: ${d}/ not found"
    exit 1
  fi
done
