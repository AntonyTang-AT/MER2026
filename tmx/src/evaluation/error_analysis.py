"""错例分析 — 基于原始 openset 标签。"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from src.evaluation.mertools_bridge import load_gt_from_csv, parse_openset_string


@dataclass
class SampleError:
    name: str
    gt_labels: list[str]
    pred_labels: list[str]
    intersection: list[str]
    missing: list[str]
    extra: list[str]
    jaccard: float


@dataclass
class ErrorSummary:
    num_samples: int
    zero_overlap: int
    avg_jaccard: float
    top_missing_labels: list[tuple[str, int]]
    top_extra_labels: list[tuple[str, int]]
    worst_samples: list[SampleError]


def _normalize_labels(text: str) -> list[str]:
    labels = parse_openset_string(text)
    return sorted({str(x).lower().strip() for x in labels if str(x).strip()})


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def analyze_errors(
    name2pred: dict[str, str],
    *,
    gt_csv: Path | str | None = None,
    name2gt: dict[str, str] | None = None,
    process_names: list[str] | None = None,
    top_n: int = 20,
) -> ErrorSummary:
    if name2gt is None:
        if gt_csv is None:
            raise ValueError("Either gt_csv or name2gt must be provided")
        name2gt = load_gt_from_csv(gt_csv)

    if process_names is None:
        process_names = list(name2gt.keys())

    sample_errors: list[SampleError] = []
    missing_counter: Counter[str] = Counter()
    extra_counter: Counter[str] = Counter()
    zero_overlap = 0
    jaccards: list[float] = []

    for name in process_names:
        gt_text = name2gt.get(name, "")
        pred_text = name2pred.get(name, "")
        gt_labels = _normalize_labels(gt_text)
        pred_labels = _normalize_labels(pred_text)
        gt_set = set(gt_labels)
        pred_set = set(pred_labels)
        inter = sorted(gt_set & pred_set)
        missing = sorted(gt_set - pred_set)
        extra = sorted(pred_set - gt_set)
        jac = _jaccard(gt_set, pred_set)
        jaccards.append(jac)
        if not inter and (gt_set or pred_set):
            zero_overlap += 1
        for label in missing:
            missing_counter[label] += 1
        for label in extra:
            extra_counter[label] += 1
        sample_errors.append(
            SampleError(
                name=name,
                gt_labels=gt_labels,
                pred_labels=pred_labels,
                intersection=inter,
                missing=missing,
                extra=extra,
                jaccard=jac,
            )
        )

    worst = sorted(sample_errors, key=lambda s: (s.jaccard, len(s.missing)))[:top_n]
    avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0

    return ErrorSummary(
        num_samples=len(process_names),
        zero_overlap=zero_overlap,
        avg_jaccard=avg_jaccard,
        top_missing_labels=missing_counter.most_common(20),
        top_extra_labels=extra_counter.most_common(20),
        worst_samples=worst,
    )


def write_error_report(summary: ErrorSummary, out_prefix: Path) -> tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")

    payload = {
        "num_samples": summary.num_samples,
        "zero_overlap": summary.zero_overlap,
        "avg_jaccard": summary.avg_jaccard,
        "top_missing_labels": summary.top_missing_labels,
        "top_extra_labels": summary.top_extra_labels,
        "worst_samples": [asdict(s) for s in summary.worst_samples],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["name", "jaccard", "gt_labels", "pred_labels", "missing", "extra"]
        )
        for s in summary.worst_samples:
            writer.writerow(
                [
                    s.name,
                    f"{s.jaccard:.4f}",
                    ",".join(s.gt_labels),
                    ",".join(s.pred_labels),
                    ",".join(s.missing),
                    ",".join(s.extra),
                ]
            )

    return json_path, csv_path
