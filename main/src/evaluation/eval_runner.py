"""批量评估入口 — EW-F1 + 可选错例分析。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.core.config_loader import get_project_root, load_global_config
from src.data.human_ov_split import load_train_names, load_val_names
from src.evaluation.error_analysis import analyze_errors, write_error_report
from src.evaluation.ew_metric import compute_ew_f1
from src.evaluation.mertools_bridge import load_gt_from_csv, load_npz_predictions


def _default_gt_csv() -> Path:
    cfg = load_global_config()
    return (get_project_root() / cfg["paths"]["data_root"] / "track2_train_human.csv").resolve()


def _resolve_process_names(split: str) -> list[str] | None:
    if split == "all":
        return None
    if split == "val":
        return load_val_names()
    if split == "train":
        return load_train_names()
    raise ValueError(f"Unknown split: {split}")


def run_eval(
    pred_path: Path,
    *,
    gt_csv: Path | None = None,
    split: str = "all",
    analyze: bool = False,
    out: Path | None = None,
) -> dict:
    gt_csv = gt_csv or _default_gt_csv()
    name2pred = load_npz_predictions(pred_path)
    name2gt = load_gt_from_csv(gt_csv)
    process_names = _resolve_process_names(split)

    result = compute_ew_f1(
        name2pred,
        name2gt=name2gt,
        process_names=process_names,
    )

    payload: dict = {
        "pred": str(pred_path),
        "gt_csv": str(gt_csv),
        "split": split,
        "num_pred": len(name2pred),
        "num_eval": len(process_names) if process_names else len(name2gt),
        **result.as_dict(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if analyze:
        summary = analyze_errors(
            name2pred,
            name2gt=name2gt,
            process_names=process_names,
        )
        payload["error_summary"] = {
            "zero_overlap": summary.zero_overlap,
            "avg_jaccard": summary.avg_jaccard,
            "top_missing_labels": summary.top_missing_labels[:10],
            "top_extra_labels": summary.top_extra_labels[:10],
        }
        logs_root = (
            get_project_root()
            / load_global_config()["paths"]["outputs_root"]
            / "eval_logs"
        )
        logs_root.mkdir(parents=True, exist_ok=True)
        stem = out.stem if out else pred_path.stem
        prefix = logs_root / f"{stem}_errors"
        json_path, csv_path = write_error_report(summary, prefix)
        payload["error_json"] = str(json_path)
        payload["error_csv"] = str(csv_path)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"EW-F1={result.fscore * 100:.2f}%  "
        f"P={result.precision * 100:.2f}%  "
        f"R={result.recall * 100:.2f}%  "
        f"(split={split}, n={payload['num_eval']})"
    )
    if out is not None:
        print(f"Saved: {out}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Local EW-F1 evaluation")
    parser.add_argument("--pred", type=Path, required=True, help="Path to openset.npz")
    parser.add_argument("--gt-csv", type=Path, default=None, help="Ground-truth CSV")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Evaluate on subset",
    )
    parser.add_argument("--analyze", action="store_true", help="Write error analysis report")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = parser.parse_args()

    run_eval(
        args.pred,
        gt_csv=args.gt_csv,
        split=args.split,
        analyze=args.analyze,
        out=args.out,
    )


if __name__ == "__main__":
    main()
