"""CodaBench 提交文件格式化。"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.config_loader import get_project_root, load_global_config
from src.evaluation.mertools_bridge import load_npz_predictions


def _candidate_csv() -> Path:
    cfg = load_global_config()
    return (
        get_project_root() / cfg["paths"]["data_root"] / "track1_track2_candidate.csv"
    ).resolve()


def _default_out_dir() -> Path:
    cfg = load_global_config()
    return (get_project_root() / cfg["paths"]["outputs_root"] / "submissions").resolve()


def format_submission(
    name2pred: dict[str, str],
    *,
    candidate_csv: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    candidate_csv = candidate_csv or _candidate_csv()
    if not candidate_csv.is_file():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")

    df = pd.read_csv(candidate_csv)
    names = [str(x) for x in df["name"].tolist()]
    expected = 20000
    if len(names) != expected:
        warnings.warn(f"Expected {expected} candidate names, got {len(names)}")

    missing = 0
    rows: list[dict[str, str]] = []
    for name in names:
        pred = name2pred.get(name, "")
        if name not in name2pred:
            missing += 1
        rows.append({"name": name, "openset": pred})

    if missing:
        warnings.warn(f"{missing} candidate samples have no prediction (empty openset)")

    out_df = pd.DataFrame(rows, columns=["name", "openset"])
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _default_out_dir() / f"track2_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Format CodaBench Track2 submission CSV")
    parser.add_argument("--pred", type=Path, required=True, help="Path to openset.npz")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--candidate-csv", type=Path, default=None)
    args = parser.parse_args()

    name2pred = load_npz_predictions(args.pred)
    out = format_submission(
        name2pred,
        candidate_csv=args.candidate_csv,
        out_path=args.out,
    )
    print(f"Submission written: {out} ({sum(1 for _ in open(out)) - 1} rows)")


if __name__ == "__main__":
    main()
