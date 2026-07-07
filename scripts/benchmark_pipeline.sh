#!/usr/bin/env bash
# Pipeline 性能基准（默认 eval + limit 100，跑 3 次）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

LIMIT=100
MODE=eval
RUNS=3
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

OUT_DIR="${ROOT}/outputs/pipeline_runs"
REPORT="${ROOT}/docs/reports/pipeline_benchmark.md"
mkdir -p "${OUT_DIR}"

echo "Benchmark: mode=${MODE} limit=${LIMIT} runs=${RUNS}"

TIMINGS=()
for i in $(seq 1 "${RUNS}"); do
  echo "--- run ${i}/${RUNS} ---"
  python -m src.inference.pipeline \
    --mode "${MODE}" \
    --limit "${LIMIT}" \
    --dry-run \
    --profile \
    --skip-stages affectgpt,openset \
    "${EXTRA[@]}"
  LATEST="$(ls -t "${OUT_DIR}"/timing_${MODE}_*.json 2>/dev/null | head -1 || true)"
  if [[ -n "${LATEST}" ]]; then
    TOTAL="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${LATEST}").read_text())
print(data.get("timing", {}).get("total", 0))
PY
)"
    TIMINGS+=("${TOTAL}")
  fi
done

python - <<PY
from pathlib import Path
import statistics

timings = [float(x) for x in "${TIMINGS[*]}".split() if x.strip()]
report = Path("${REPORT}")
report.parent.mkdir(parents=True, exist_ok=True)

lines = [
    "# Pipeline Benchmark",
    "",
    f"- mode: ${MODE}",
    f"- limit: ${LIMIT}",
    f"- runs: ${RUNS}",
    f"- dry_run: true (skip affectgpt/openset GPU stages)",
    "",
    "## Timing (seconds)",
    "",
]
if timings:
    mean = statistics.mean(timings)
    stdev = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    lines.append(f"- total mean: {mean:.3f}s")
    lines.append(f"- total stdev: {stdev:.3f}s")
    lines.append(f"- samples: {timings}")
else:
    lines.append("- no timing captured")

report.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"Report written: {report}")
PY
