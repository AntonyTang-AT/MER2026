#!/usr/bin/env bash
# 创建 conda 环境（需已安装 conda）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/third_party/MERTools/MER2026/environment_vllm3.yml"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Run scripts/clone_mertools.sh first."
  exit 1
fi

conda env create -f "${ENV_FILE}" || conda env update -f "${ENV_FILE}"
echo "Activate: conda activate vllm3"
