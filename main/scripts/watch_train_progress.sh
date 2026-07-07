#!/usr/bin/env bash
# 实时训练进度监控（包装 watch_train_progress.py）
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${1:-${ROOT}/logs/train_human_full.log}"
exec python3 "${ROOT}/scripts/watch_train_progress.py" --log "${LOG}"
