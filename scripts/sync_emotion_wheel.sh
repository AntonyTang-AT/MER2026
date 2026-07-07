#!/usr/bin/env bash
# 从 MERTools upstream 同步 emotion_wheel 评估资源
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/third_party/MERTools/MER2026/MER2026_Track2/emotion_wheel"
BASE="https://raw.githubusercontent.com/zeroQiaoba/MERTools/master/MER2026/MER2026_Track2/emotion_wheel"

mkdir -p "${DEST}"

FILES=(
  wheel_mapping.npz
  synonym.xlsx
  wheel1.xlsx
  wheel2.xlsx
  wheel3.xlsx
  wheel4.xlsx
  wheel5.xlsx
)

for f in "${FILES[@]}"; do
  out="${DEST}/${f}"
  if [[ -f "${out}" ]] && [[ -s "${out}" ]]; then
    echo "skip (exists): ${f}"
    continue
  fi
  echo "download: ${f}"
  curl -fsSL "${BASE}/${f}" -o "${out}"
done

echo "========== verify wheel_mapping.npz =========="
python3 << PY
import sys
from pathlib import Path
import numpy as np

path = Path("${DEST}") / "wheel_mapping.npz"
if not path.is_file():
    print("FAIL: wheel_mapping.npz missing")
    sys.exit(1)
data = np.load(path, allow_pickle=True)
for key in ("format_mapping", "raw_mapping", "wheel_map_whole"):
    if key not in data:
        print(f"FAIL: missing key {key}")
        sys.exit(1)
print("OK: wheel_mapping.npz keys verified")
PY

echo "emotion_wheel sync complete: ${DEST}"
