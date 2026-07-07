#!/usr/bin/env bash
# 克隆官方 MERTools 到 third_party/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${ROOT}/third_party/MERTools"

if [[ -d "${TARGET}/.git" ]]; then
  echo "MERTools already cloned at ${TARGET}"
  exit 0
fi

mkdir -p "${ROOT}/third_party"
git clone --depth 1 https://github.com/zeroQiaoba/MERTools.git "${TARGET}"
echo "Cloned to ${TARGET}"
echo "Track2 code: ${TARGET}/MER2026/MER2026_Track2"
