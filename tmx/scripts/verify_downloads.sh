#!/usr/bin/env bash
# 校验 MER2026 数据集与模型完整性，失败 exit 1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/mer2026-dataset"
MODELS="${ROOT}/models"
FAIL=0

err() { echo "FAIL: $*"; FAIL=1; }
ok()  { echo "OK: $*"; }

echo "========== CSV =========="
for f in track2_train_human.csv track1_track2_candidate.csv subtitle_chieng.csv; do
  [[ -f "${DEST}/${f}" ]] && ok "${f}" || err "missing ${f}"
done
[[ $(wc -l < "${DEST}/track2_train_human.csv") -eq 1533 ]] && ok "track2_train_human lines=1533" || err "track2_train_human line count"
[[ $(wc -l < "${DEST}/track1_track2_candidate.csv") -eq 20001 ]] && ok "track1_track2_candidate lines=20001" || err "track1_track2_candidate line count"

echo "========== ZIP (9) =========="
ZIPS=(
  "audio_7z/audio_track2_train_human/audio_split.zip"
  "audio_7z/audio_track1_track2_candidate/audio_split_0001.zip"
  "audio_7z/audio_track1_track2_candidate/audio_split_0002.zip"
  "video_7z/video_track2_train_human/video_split.zip"
  "video_7z/video_track1_track2_candidate/video_split_0001.zip"
  "video_7z/video_track1_track2_candidate/video_split_0002.zip"
  "openface_7z/openface_track2_train_human/openface_split.zip"
  "openface_7z/openface_track1_track2_candidate/openface_split_0001.zip"
  "openface_7z/openface_track1_track2_candidate/openface_split_0002.zip"
)
for z in "${ZIPS[@]}"; do
  f="${DEST}/${z}"
  if [[ -f "$f" ]] && 7z t "$f" >/dev/null 2>&1; then
    ok "zip ${z} ($(du -h "$f" | cut -f1))"
  else
    err "zip ${z}"
  fi
done

echo "========== EXTRACTED DIRS =========="
for d in audio video openface_face; do
  if [[ -d "${DEST}/${d}" ]]; then
    n=$(find "${DEST}/${d}" -type f | wc -l)
    (( n > 0 )) && ok "${d}/ ${n} files" || err "${d}/ empty"
  else
    err "${d}/ not found"
  fi
done

echo "========== MODELS =========="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

python3 << PY
import os, sys, json
from pathlib import Path

fail = False
def e(msg):
    global fail
    print(f"FAIL: {msg}")
    fail = True
def o(msg):
    print(f"OK: {msg}")

models = Path("${MODELS}")

# CLIP
clip = models / "clip-vit-large-patch14" / "pytorch_model.bin"
if clip.is_file() and clip.stat().st_size == 1710671599:
    o(f"CLIP bin {clip.stat().st_size}")
else:
    e(f"CLIP bin size {clip.stat().st_size if clip.is_file() else 'missing'}")

# HuBERT
hub = models / "chinese-hubert-large" / "pytorch_model.bin"
if hub.is_file() and hub.stat().st_size > 1_000_000_000:
    o(f"HuBERT bin {hub.stat().st_size}")
else:
    e(f"HuBERT bin")

# Qwen shards
qdir = models / "Qwen2.5-7B-Instruct"
idx = qdir / "model.safetensors.index.json"
if not idx.is_file():
    e("Qwen index missing")
else:
    shards = sorted(set(json.load(open(idx))["weight_map"].values()))
    for s in shards:
        p = qdir / s
        if p.is_file() and p.stat().st_size > 0:
            o(f"Qwen {s} {p.stat().st_size}")
        else:
            e(f"Qwen {s}")

# load test
try:
    from transformers import CLIPModel, HubertModel, AutoModelForCausalLM, AutoTokenizer
    import torch
    CLIPModel.from_pretrained(str(models/"clip-vit-large-patch14"), local_files_only=True)
    o("CLIP load")
    HubertModel.from_pretrained(str(models/"chinese-hubert-large"), local_files_only=True)
    o("HuBERT load")
    AutoModelForCausalLM.from_pretrained(str(qdir), local_files_only=True, torch_dtype=torch.float16, device_map="cpu")
    o("Qwen load")
except Exception as ex:
    e(f"model load: {ex}")

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
