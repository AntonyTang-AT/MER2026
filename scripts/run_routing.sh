#!/usr/bin/env bash
# L4 批量路由（CPU，可与训练并行）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

SPLIT="${1:-human}"
shift || true

LIMIT=""
CHECK=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --check-media) CHECK="--check-media"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

ARGS=(--split "${SPLIT}")
[[ -n "${LIMIT}" ]] && ARGS+=(--limit "${LIMIT}")
[[ -n "${CHECK}" ]] && ARGS+=("${CHECK}")
ARGS+=("${EXTRA[@]}")

python -m src.routing.run_routing "${ARGS[@]}"
