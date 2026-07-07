"""Baseline 评估报告 — experiments/exp001_baseline。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.core.config_loader import get_project_root, load_global_config, load_yaml
from src.data.human_ov_split import load_val_names
from src.evaluation.ew_metric import compute_ew_f1
from src.evaluation.error_analysis import analyze_errors, write_error_report
from src.evaluation.mertools_bridge import load_gt_from_csv, load_npz_predictions


@dataclass
class BaselineRecord:
    model: str
    dataset: str
    ew_f1: float
    precision: float
    recall: float
    split: str
    pred_npz: str
    notes: str = ""


def exp001_dir() -> Path:
    return get_project_root() / "experiments" / "exp001_baseline"


def evaluate_openset_npz(
    pred_npz: Path,
    *,
    split: str = "val",
    model_name: str = "affectgpt_human",
    dataset: str = "human_ov",
) -> BaselineRecord:
    global_cfg = load_global_config()
    gt_csv = get_project_root() / global_cfg["paths"]["data_root"] / "track2_train_human.csv"
    name2pred = load_npz_predictions(pred_npz)
    process_names = load_val_names() if split == "val" else None

    result = compute_ew_f1(
        name2pred,
        gt_csv=gt_csv,
        process_names=process_names,
    )

    summary = analyze_errors(
        name2pred,
        gt_csv=gt_csv,
        process_names=process_names,
    )
    logs = get_project_root() / global_cfg["paths"]["outputs_root"] / "eval_logs"
    write_error_report(summary, logs / f"exp001_{pred_npz.stem}_errors")

    return BaselineRecord(
        model=model_name,
        dataset=dataset,
        ew_f1=result.fscore,
        precision=result.precision,
        recall=result.recall,
        split=split,
        pred_npz=str(pred_npz),
    )


def write_exp001_report(records: list[BaselineRecord], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or exp001_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines = [
        "# exp001: 官方 AffectGPT Baseline 复现",
        "",
        f"- 更新时间: {ts}",
        f"- 目标: val EW-F1 ≥ 59%",
        "",
        "## 结果",
        "",
        "| 模型 | 数据 | Split | EW-F1 | Precision | Recall | 预测文件 | 备注 |",
        "|------|------|-------|-------|-----------|--------|----------|------|",
    ]
    for r in records:
        md_lines.append(
            f"| {r.model} | {r.dataset} | {r.split} | "
            f"{r.ew_f1 * 100:.2f}% | {r.precision * 100:.2f}% | {r.recall * 100:.2f}% | "
            f"`{Path(r.pred_npz).name}` | {r.notes} |"
        )

    md_path = out_dir / "RESULTS.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return md_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate exp001 baseline report")
    parser.add_argument("--pred", type=Path, required=True, help="openset.npz path")
    parser.add_argument("--split", default="val", choices=["val", "train", "all"])
    parser.add_argument("--model-name", default="affectgpt_human")
    args = parser.parse_args()

    record = evaluate_openset_npz(
        args.pred,
        split=args.split,
        model_name=args.model_name,
    )
    md = write_exp001_report([record])
    print(
        f"EW-F1={record.ew_f1 * 100:.2f}%  "
        f"P={record.precision * 100:.2f}%  R={record.recall * 100:.2f}%"
    )
    print(f"Report: {md}")


if __name__ == "__main__":
    main()
