#!/usr/bin/env bash
# DeepSeek V4 Pro：分歧子集 reason 改写 → official openset → 生产栈 → 可提交 zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3
set -a
# strip CR from .env values
source <(sed 's/\r$//' "${ROOT}/.env")
set +a

export REASON_API_KEY="${REASON_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export REASON_API_KEY="$(printf '%s' "${REASON_API_KEY}" | tr -d '\r\n')"
export REASON_API_BASE_URL="${REASON_API_BASE_URL:-https://api.deepseek.com}"
export REASON_API_MODEL="${REASON_API_MODEL:-deepseek-v4-pro}"

IN_REASON="${IN_REASON:-${ROOT}/third_party/MERTools/MER2026/MER2026_Track2/output/results-mer2026ov-rl-e3-mer2026ov/human_outputhybird_bestsetup_bestfusion_face_lz_20260712171/checkpoint_000003_exp014_rl_candidate.npz}"
NAMES_JSON="${NAMES_JSON:-${ROOT}/outputs/exp015/divergent_samples_candidate20k.json}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/reason_api}"
TAG="${TAG:-deepseek_v4pro_div4280}"
OUT_REASON="${OUT_REASON:-${OUT_DIR}/rl_reason_${TAG}.npz}"
OUT_OPENSET_SUB="${OUT_OPENSET_SUB:-${OUT_DIR}/rl_openset_${TAG}.npz}"
OUT_OPENSET_FULL="${OUT_OPENSET_FULL:-${OUT_DIR}/RL_v2_e3_${TAG}_candidate20k.npz}"
OUT_STACK="${OUT_STACK:-${OUT_DIR}/R3_RL_triple_ser_lr_dtrb_${TAG}_candidate20k.npz}"
OUT_CSV="${OUT_CSV:-${ROOT}/outputs/submissions/R3_RL_triple_ser_lr_dtrb_${TAG}_candidate20k.csv}"
OUT_ZIP="${OUT_ZIP:-${ROOT}/outputs/submissions/R3_RL_triple_ser_lr_dtrb_${TAG}_candidate20k.zip}"
BASE_OPENSET="${BASE_OPENSET:-${ROOT}/outputs/exp014/RL_v2_e3_candidate20k.npz}"
E14="${E14:-${ROOT}/outputs/exp014/SFT_v3_e14_candidate20k.npz}"
E15="${E15:-${ROOT}/outputs/exp014/SFT_v3_e15_candidate20k.npz}"
ROUTER="${ROUTER:-${ROOT}/outputs/exp015/ser_router_model.json}"
CHUNK="${CHUNK:-100}"
CONCURRENCY="${CONCURRENCY:-8}"
CUDA="${CUDA:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
PROGRESS="${PROGRESS:-${ROOT}/logs/reason_api_${TAG}_pipeline.log}"

mkdir -p "${OUT_DIR}" "${ROOT}/outputs/submissions" "${ROOT}/logs"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS}"; }

PROMPT_VARIANT="${PROMPT_VARIANT:-refine_v1}"
export ROOT IN_REASON NAMES_JSON OUT_DIR TAG OUT_REASON OUT_OPENSET_SUB OUT_OPENSET_FULL \
  OUT_STACK OUT_CSV OUT_ZIP BASE_OPENSET E14 E15 ROUTER CHUNK CONCURRENCY CUDA \
  VLLM_GPU_MEMORY_UTILIZATION PROGRESS REASON_API_MODEL REASON_API_KEY REASON_API_BASE_URL \
  PROMPT_VARIANT

log "START tag=${TAG} model=${REASON_API_MODEL} variant=${PROMPT_VARIANT} concurrency=${CONCURRENCY}"

# ---- 1) chunked reason refine (resume-safe) ----
python -u - <<'PY' 2>&1 | tee -a "${PROGRESS}"
import json, os, time
from pathlib import Path
from src.inference.reason_api import load_reason_api_config, load_reason_map, save_reason_map, load_names_json
from src.inference.reason_api.provider import open_provider

root = Path(os.environ.get("ROOT", "."))
in_reason = Path(os.environ["IN_REASON"])
out_reason = Path(os.environ["OUT_REASON"])
names_json = Path(os.environ["NAMES_JSON"])
chunk = int(os.environ.get("CHUNK", "100"))
concurrency = int(os.environ.get("CONCURRENCY", "8"))
model = os.environ.get("REASON_API_MODEL", "deepseek-v4-pro")

base = load_reason_map(in_reason)
subset = set(load_names_json(names_json))
target = [n for n in sorted(base.keys()) if n in subset]
print(f"[refine] target={len(target)} total_base={len(base)} chunk={chunk}", flush=True)

out = dict(base)
if out_reason.is_file():
    prev = load_reason_map(out_reason)
    out.update(prev)
    print(f"[refine] resumed from {out_reason} keys={len(prev)}", flush=True)

# already refined = different from base (or marked progress file)
done_path = out_reason.with_suffix(".done.json")
done = set()
if done_path.is_file():
    done = set(json.loads(done_path.read_text()))
# also treat changed text as done
for n in target:
    if n in out and out[n] != base.get(n, "") and n not in done:
        done.add(n)

pending = [n for n in target if n not in done]
print(f"[refine] done={len(done)} pending={len(pending)}", flush=True)

cfg = load_reason_api_config()
cfg.model = model
cfg.base_url = os.environ.get("REASON_API_BASE_URL", cfg.base_url)
cfg.prompt_variant = os.environ.get("PROMPT_VARIANT", cfg.prompt_variant or "refine_v1")
cfg.max_concurrency = concurrency
cfg.provider = "openai_compatible"
cfg.max_tokens = int(os.environ.get("REASON_API_MAX_TOKENS", str(cfg.max_tokens or 512)))
cfg.temperature = float(os.environ.get("REASON_API_TEMPERATURE", str(cfg.temperature or 0.2)))
cfg.timeout_s = float(os.environ.get("REASON_API_TIMEOUT_S", str(cfg.timeout_s or 120)))
print(f"[refine] base_url={cfg.base_url} variant={cfg.prompt_variant} temp={cfg.temperature}", flush=True)

t0 = time.time()
with open_provider(cfg) as api:
    for i in range(0, len(pending), chunk):
        batch = pending[i : i + chunk]
        bt = time.time()
        refined = api.refine_map(base, names=batch)
        for n in batch:
            out[n] = refined[n]
            done.add(n)
        save_reason_map(out, out_reason)
        done_path.write_text(json.dumps(sorted(done)))
        rate = len(batch) / max(time.time() - bt, 1e-6)
        eta = (len(pending) - i - len(batch)) / max(rate, 1e-6)
        print(
            f"[refine] batch {i//chunk+1}/{(len(pending)+chunk-1)//chunk} "
            f"n={len(batch)} rate={rate:.2f}/s eta_s={eta:.0f} changed="
            f"{sum(1 for n in batch if out[n]!=base[n])}",
            flush=True,
        )

save_reason_map(out, out_reason)
report = {
    "in_reason": str(in_reason),
    "out_reason": str(out_reason),
    "model": model,
    "n_selected": len(target),
    "n_done": len(done),
    "n_changed": sum(1 for n in target if out.get(n, "") != base.get(n, "")),
    "elapsed_s": round(time.time() - t0, 1),
}
Path(str(out_reason) + ".report.json").write_text(json.dumps(report, indent=2))
print("[refine] DONE", json.dumps(report), flush=True)
PY

log "REASON done -> ${OUT_REASON}"

# ---- 2) official openset on divergent subset ----
log "OPENSET start cuda=${CUDA}"
CUDA_VISIBLE_DEVICES="${CUDA}" VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
  python -u -m src.inference.openset_extractor \
    --reason-npz "${OUT_REASON}" \
    --store-npz "${OUT_OPENSET_SUB}" \
    --prompt-variant official \
    --postprocess official \
    --names-json "${NAMES_JSON}" \
    --cuda "${CUDA}" \
    2>&1 | tee -a "${PROGRESS}"
log "OPENSET subset done -> ${OUT_OPENSET_SUB}"

# ---- 3) merge openset into full RL official + rebuild stack + zip ----
python -u - <<'PY' 2>&1 | tee -a "${PROGRESS}"
import json, zipfile, csv, io, shutil
from pathlib import Path
from collections import Counter
from src.evaluation.mertools_bridge import load_npz_predictions, parse_openset_string
from src.inference.openset_postprocess import sanitize_openset_string
from src.inference.rl_openset_bridge import BridgeConfig, bridge_rl_openset
from src.inference.dtrb_boost import DTRBConfig, apply_dtrb_boost
from src.inference.expert_router import apply_routing, load_router_model
from src.inference.recall_boost import load_reason_map
from src.inference.ensemble_runner import save_merged_predictions
from src.inference.reason_api import load_names_json
import os, subprocess

base_os = load_npz_predictions(os.environ["BASE_OPENSET"])
sub_os = load_npz_predictions(os.environ["OUT_OPENSET_SUB"])
names = load_names_json(os.environ["NAMES_JSON"])
merged = dict(base_os)
n_patch = 0
for n in names:
    if n in sub_os:
        merged[n] = sub_os[n]
        n_patch += 1
out_full = Path(os.environ["OUT_OPENSET_FULL"])
save_merged_predictions(merged, out_full)
print(f"[merge] patched_openset={n_patch} -> {out_full}", flush=True)

reasons = load_reason_map(os.environ["OUT_REASON"])
bridged, _ = bridge_rl_openset(
    merged, reasons,
    cfg=BridgeConfig(mode="gap_add", max_add=1, allow_noise_swap=True, require_gap=True),
)
e14 = load_npz_predictions(os.environ["E14"])
e15 = load_npz_predictions(os.environ["E15"])
model = load_router_model(os.environ["ROUTER"])
c20k = sorted(set(bridged) & set(e14) & set(e15))
ser, _ = apply_routing(
    bridged, e14, e15, c20k,
    strategy="ser_lr", model=model,
    confidence_threshold=0.65, max_switch_rate=0.10,
)
preds, stats = apply_dtrb_boost(
    ser, bridged, e14, e15,
    cfg=DTRBConfig(reason_guided=True),
    names=c20k, reasons=reasons,
)
out_stack = Path(os.environ["OUT_STACK"])
save_merged_predictions(preds, out_stack)
print(f"[stack] n={len(preds)} dtrb_stats_keys={list(stats)[:8] if isinstance(stats, dict) else type(stats)} -> {out_stack}", flush=True)

csv_path = Path(os.environ["OUT_CSV"])
zip_path = Path(os.environ["OUT_ZIP"])
subprocess.run(
    ["python", "-m", "src.data.submission_formatter", "--pred", str(out_stack), "--out", str(csv_path), "--sanitize-mode", "ew"],
    check=True,
)
# zip answer.csv
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(csv_path, arcname="answer.csv")

# vs production
prod_zip = Path("outputs/submissions/R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8_candidate20k.zip")
with zipfile.ZipFile(prod_zip) as z:
    prod = {r["name"]: r["openset"] for r in csv.DictReader(io.StringIO(z.read("answer.csv").decode()))}
with zipfile.ZipFile(zip_path) as z:
    neu = {r["name"]: r["openset"] for r in csv.DictReader(io.StringIO(z.read("answer.csv").decode()))}

def labs(s):
    return {x.lower().strip() for x in parse_openset_string(s) if str(x).strip() and x.lower() != "neutral"}

str_chg = set_chg = 0
adds, dels = Counter(), Counter()
for n in sorted(neu):
    if neu[n] != prod.get(n, ""):
        str_chg += 1
    sp, sq = labs(neu[n]), labs(prod.get(n, "[]"))
    if sp != sq:
        set_chg += 1
        for x in sp - sq:
            adds[x] += 1
        for x in sq - sp:
            dels[x] += 1

summary = {
    "tag": os.environ.get("TAG"),
    "n_samples": len(neu),
    "str_chg_vs_prod": str_chg,
    "set_chg_vs_prod": set_chg,
    "top_adds": adds.most_common(12),
    "top_dels": dels.most_common(12),
    "out_reason": os.environ["OUT_REASON"],
    "out_stack": str(out_stack),
    "out_zip": str(zip_path),
}
Path(os.environ["OUT_DIR"], f"{os.environ['TAG']}_vs_prod.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("[summary]", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
print(f"[zip] {zip_path}", flush=True)
PY

log "ALL DONE zip=${OUT_ZIP}"
