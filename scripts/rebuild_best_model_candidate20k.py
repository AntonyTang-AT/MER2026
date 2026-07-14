#!/usr/bin/env python3
"""重建当前最优生产栈并打包提交 zip。

变体：R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8
Test EW-F1：65.7297%

流水线：RL official → RRB-gap1 → SER-lr → DTRB(reason_guided) → EW sanitize → zip

示例::

  python scripts/rebuild_best_model_candidate20k.py
  python scripts/rebuild_best_model_candidate20k.py --skip-zip
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.mertools_bridge import load_npz_predictions
from src.inference.dtrb_boost import DTRBConfig, apply_dtrb_boost
from src.inference.ensemble_runner import save_merged_predictions
from src.inference.expert_router import apply_routing, load_router_model
from src.inference.recall_boost import load_reason_map
from src.inference.rl_openset_bridge import BridgeConfig, bridge_rl_openset

RL_OPENSET = "outputs/exp014/RL_v2_e3_candidate20k.npz"
E14 = "outputs/exp014/SFT_v3_e14_candidate20k.npz"
E15 = "outputs/exp014/SFT_v3_e15_candidate20k.npz"
ROUTER = "outputs/exp015/ser_router_model.json"
REASON = (
    "third_party/MERTools/MER2026/MER2026_Track2/output/"
    "results-mer2026ov-rl-e3-mer2026ov/"
    "human_outputhybird_bestsetup_bestfusion_face_lz_20260712171/"
    "checkpoint_000003_exp014_rl_candidate.npz"
)
OUT_NPZ = "outputs/exp021/R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8_candidate20k.npz"
OUT_CSV = "outputs/submissions/R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8_candidate20k.csv"
OUT_ZIP = "outputs/submissions/R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8_candidate20k.zip"
TAG = "R3_RL_triple_ser_lr_dtrb_dtrb_reason_cap8"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild best production stack")
    parser.add_argument("--skip-zip", action="store_true")
    args = parser.parse_args()

    root = _ROOT
    rl = load_npz_predictions(root / RL_OPENSET)
    e14 = load_npz_predictions(root / E14)
    e15 = load_npz_predictions(root / E15)
    reasons = load_reason_map(root / REASON)
    bridged, _ = bridge_rl_openset(
        rl,
        reasons,
        cfg=BridgeConfig(
            mode="gap_add",
            max_add=1,
            allow_noise_swap=True,
            require_gap=True,
        ),
    )
    model = load_router_model(root / ROUTER)
    names = sorted(set(bridged) & set(e14) & set(e15))
    ser, _ = apply_routing(
        bridged,
        e14,
        e15,
        names,
        strategy="ser_lr",
        model=model,
        confidence_threshold=0.65,
        max_switch_rate=0.10,
    )
    preds, stats = apply_dtrb_boost(
        ser,
        bridged,
        e14,
        e15,
        cfg=DTRBConfig(reason_guided=True),
        names=names,
        reasons=reasons,
    )
    out_npz = root / OUT_NPZ
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_merged_predictions(preds, out_npz)
    print(f"[npz] {out_npz} n={len(preds)} stats={stats if isinstance(stats, dict) else type(stats)}")

    if args.skip_zip:
        return

    out_csv = root / OUT_CSV
    out_zip = root / OUT_ZIP
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data.submission_formatter",
            "--pred",
            str(out_npz),
            "--out",
            str(out_csv),
            "--sanitize-mode",
            "ew",
        ],
        check=True,
        cwd=str(root),
    )
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname="answer.csv")
    print(f"[zip] {out_zip} tag={TAG}")


if __name__ == "__main__":
    main()
