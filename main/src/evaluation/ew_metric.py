"""封装 MERTools wheel.py — EW-F1 评估。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.evaluation.mertools_bridge import import_wheel, load_gt_from_csv, mertools_context


@dataclass
class EvalResult:
    fscore: float
    precision: float
    recall: float

    @property
    def ew_f1(self) -> float:
        return self.fscore

    def as_dict(self) -> dict[str, float]:
        return {
            "ew_f1": self.fscore,
            "precision": self.precision,
            "recall": self.recall,
        }


def compute_ew_f1(
    name2pred: dict[str, str],
    *,
    gt_csv: Path | str | None = None,
    name2gt: dict[str, str] | None = None,
    process_names: list[str] | None = None,
) -> EvalResult:
    """计算官方 EW-F1（5 wheel level1 平均）。"""
    if name2gt is None:
        if gt_csv is None:
            raise ValueError("Either gt_csv or name2gt must be provided")
        name2gt = load_gt_from_csv(gt_csv)

    wheel_metric_calculation, _ = import_wheel()
    with mertools_context():
        fscore, precision, recall = wheel_metric_calculation(
            name2gt=name2gt,
            name2pred=name2pred,
            process_names=process_names,
            inter_print=False,
            level="level1",
        )

    return EvalResult(fscore=float(fscore), precision=float(precision), recall=float(recall))
