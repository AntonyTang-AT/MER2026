"""EW-F1 metric tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.ew_metric import compute_ew_f1
from src.core.config_loader import get_project_root


def _gt_as_pred_npz(gt_csv: Path, out: Path) -> None:
    df = pd.read_csv(gt_csv)
    names = df["name"].astype(str).tolist()
    opensets = df["openset"].astype(str).tolist()
    np.savez_compressed(out, filenames=names, fileitems=opensets)


def test_gt_as_pred_ew_f1_near_one():
    gt_csv = get_project_root() / "data/mer2026-dataset/track2_train_human.csv"
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = Path(tmp) / "gt_pred.npz"
        _gt_as_pred_npz(gt_csv, npz_path)

        df = pd.read_csv(gt_csv)
        name2pred = {
            str(row["name"]): str(row["openset"]) for _, row in df.iterrows()
        }
        result = compute_ew_f1(name2pred, gt_csv=gt_csv)
        assert result.fscore > 0.99
