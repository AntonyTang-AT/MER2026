#!/usr/bin/env bash
# 生成 CodaBench 提交文件 — 阶段 1.5 / 8 完成后启用
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "TODO: python -m src.data.submission_formatter"
echo "Output: ${ROOT}/outputs/submissions/"
