#!/usr/bin/env python3
"""训练 SER 路由模型（val306）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.config_loader import get_project_root, load_global_config
from src.data.human_ov_split import load_val_names
from src.evaluation.mertools_bridge import load_gt_from_csv, load_npz_predictions, parse_openset_string
from src.inference.expert_router import save_router_model, train_router_lr

VAL_COMPONENTS = {
    "rl": (
        "third_party/MERTools/MER2026/MER2026_Track2/output/"
        "results-mer2026ov-rlval-20260712171-human/"
        "human_outputhybird_bestsetup_bestfusion_face_lz_20260712171/"
        "checkpoint_000003_loss_0.128-openset.npz"
    ),
    "e14": (
        "third_party/MERTools/MER2026/MER2026_Track2/output/results-mer2026ov-human/"
        "human_outputhybird_bestsetup_bestfusion_face_lz_20260712025/"
        "checkpoint_000014_loss_1.479-openset.npz"
    ),
    "e15": (
        "third_party/MERTools/MER2026/MER2026_Track2/output/results-mer2026ov-human/"
        "human_outputhybird_bestsetup_bestfusion_face_lz_20260712025/"
        "checkpoint_000015_loss_1.128-openset.npz"
    ),
}


def _norm_labels(text: str) -> list[str]:
    return sorted({x.lower().strip() for x in parse_openset_string(text) if str(x).strip()})


def _subset_dict(raw: dict[str, str], names: list[str]) -> dict[str, str]:
    return {n: raw[n] for n in names if n in raw}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SER router on val306")
    parser.add_argument("--out", default="outputs/exp015/ser_router_model.json")
    args = parser.parse_args()

    root = get_project_root()
    gt_csv = root / load_global_config()["paths"]["data_root"] / "track2_train_human.csv"
    names = load_val_names()
    gt_raw = load_gt_from_csv(gt_csv)
    gt_labels = {k: _norm_labels(v) for k, v in gt_raw.items()}

    rl = _subset_dict(load_npz_predictions(root / VAL_COMPONENTS["rl"]), names)
    e14 = _subset_dict(load_npz_predictions(root / VAL_COMPONENTS["e14"]), names)
    e15 = _subset_dict(load_npz_predictions(root / VAL_COMPONENTS["e15"]), names)

    model = train_router_lr(names, rl, e14, e15, gt_labels, classes=["triple", "v3_union"])
    out_path = root / args.out
    save_router_model(model, out_path)

    meta = {
        "n_train": len(names),
        "classes": model.classes,
        "strategy": model.strategy,
        "confidence_threshold": model.confidence_threshold,
        "default_expert": model.default_expert,
        "model_path": str(out_path),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved router model: {out_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
