"""MER-Caption+ 质量过滤 — 阶段 5.1。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config_loader import get_project_root, load_global_config


@dataclass
class FilterConfig:
    drop_empty_openset: bool = True
    min_labels: int = 1
    max_labels: int = 14
    require_media: bool = False
    exclude_human_names: bool = True
    exclude_candidate_overlap: bool = False
    normalize_via_wheel: bool = True
    max_unknown_label_ratio: float = 0.5


@dataclass
class MediaRoots:
    audio: Path
    video: Path
    face: Path


@dataclass
class FilterSummary:
    input_count: int = 0
    output_count: int = 0
    removed: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_openset(raw: object) -> list[str]:
    """与官方 toolkit.utils.functions.string_to_list 行为对齐。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text[0] == "[":
        text = text[1:]
    if text and text[-1] == "]":
        text = text[:-1]
    return [item.strip() for item in re.split(r"['\",]", text) if item.strip()]


def default_data_root() -> Path:
    cfg = load_global_config()
    return (get_project_root() / cfg["paths"]["data_root"]).resolve()


def default_mercaptionplus_csv(data_root: Path | None = None) -> Path:
    root = data_root or default_data_root()
    return root / "track2_train_mercaptionplus.csv"


def default_filtered_csv(data_root: Path | None = None) -> Path:
    root = data_root or default_data_root()
    return root / "track2_train_mercaptionplus_filtered.csv"


def load_mercaptionplus_table(csv_path: Path | str) -> pd.DataFrame:
    path = Path(csv_path)
    df = pd.read_csv(path)
    required = {"name", "openset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}: {path}")
    return df


def load_name_set(csv_path: Path | str) -> set[str]:
    df = pd.read_csv(csv_path)
    if "name" not in df.columns:
        raise ValueError(f"CSV missing name column: {csv_path}")
    return set(df["name"].astype(str))


def load_known_labels_from_wheel(mertools_root: Path | None = None) -> set[str] | None:
    cfg = load_global_config()
    root = get_project_root()
    mertools = mertools_root or (root / cfg["paths"]["mertools_root"]).resolve()
    wheel = mertools / "emotion_wheel" / "wheel_mapping.npz"
    if not wheel.is_file():
        return None
    import numpy as np

    data = np.load(wheel, allow_pickle=True)
    if "format_mapping" not in data:
        return None
    mapping = data["format_mapping"].item()
    return {str(k).strip().lower() for k in mapping.keys()}


def media_paths(name: str, roots: MediaRoots) -> tuple[Path, Path, Path]:
    audio = roots.audio / f"{name}.wav"
    video = roots.video / f"{name}.mp4"
    face = roots.face / name / f"{name}.npy"
    return audio, video, face


def has_media(name: str, roots: MediaRoots) -> bool:
    audio, video, face = media_paths(name, roots)
    return audio.is_file() and video.is_file() and face.is_file()


def _unknown_label_ratio(labels: list[str], known: set[str] | None) -> float:
    if not labels:
        return 0.0
    if not known:
        return 0.0
    unknown = sum(1 for label in labels if label.strip().lower() not in known)
    return unknown / len(labels)


def apply_filters(
    df: pd.DataFrame,
    cfg: FilterConfig,
    *,
    media_roots: MediaRoots | None = None,
    human_names: set[str] | None = None,
    candidate_names: set[str] | None = None,
    known_labels: set[str] | None = None,
) -> tuple[pd.DataFrame, FilterSummary]:
    summary = FilterSummary(input_count=len(df))
    kept_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        name = str(row["name"])
        labels = parse_openset(row["openset"])
        remove_reason: str | None = None

        if cfg.drop_empty_openset and len(labels) == 0:
            remove_reason = "empty_openset"
        elif len(labels) < cfg.min_labels:
            remove_reason = "too_few_labels"
        elif len(labels) > cfg.max_labels:
            remove_reason = "too_many_labels"
        elif cfg.exclude_human_names and human_names and name in human_names:
            remove_reason = "overlap_human"
        elif cfg.exclude_candidate_overlap and candidate_names and name in candidate_names:
            remove_reason = "overlap_candidate"
        elif cfg.require_media:
            if media_roots is None:
                raise ValueError("require_media=True but media_roots is None")
            if not has_media(name, media_roots):
                remove_reason = "missing_media"
        elif cfg.normalize_via_wheel and known_labels is not None:
            ratio = _unknown_label_ratio(labels, known_labels)
            if ratio > cfg.max_unknown_label_ratio:
                remove_reason = "unknown_labels"

        if remove_reason:
            summary.removed[remove_reason] = summary.removed.get(remove_reason, 0) + 1
            continue

        kept_rows.append({"name": name, "openset": row["openset"]})

    out = pd.DataFrame(kept_rows, columns=["name", "openset"])
    summary.output_count = len(out)
    return out, summary


def write_filtered_csv(df: pd.DataFrame, out_path: Path | str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def summarize_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
    summary: FilterSummary,
    *,
    label_lengths_before: list[int] | None = None,
) -> dict[str, Any]:
    if label_lengths_before is None:
        label_lengths_before = [len(parse_openset(v)) for v in before["openset"]]
    label_lengths_after = [len(parse_openset(v)) for v in after["openset"]]

    def _hist(values: list[int]) -> dict[str, int]:
        hist: dict[str, int] = {}
        for value in values:
            key = str(value)
            hist[key] = hist.get(key, 0) + 1
        return dict(sorted(hist.items(), key=lambda item: int(item[0])))

    return {
        "input_count": summary.input_count,
        "output_count": summary.output_count,
        "removed": summary.removed,
        "label_count_hist_before": _hist(label_lengths_before),
        "label_count_hist_after": _hist(label_lengths_after),
    }


def run_filter(
    *,
    input_csv: Path | str | None = None,
    output_csv: Path | str | None = None,
    report_json: Path | str | None = None,
    cfg: FilterConfig | None = None,
    data_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = data_root or default_data_root()
    cfg = cfg or FilterConfig()
    input_path = Path(input_csv) if input_csv else default_mercaptionplus_csv(root)
    output_path = Path(output_csv) if output_csv else default_filtered_csv(root)

    before = load_mercaptionplus_table(input_path)
    human_csv = root / "track2_train_human.csv"
    candidate_csv = root / "track1_track2_candidate.csv"
    human_names = load_name_set(human_csv) if human_csv.is_file() else set()
    candidate_names = load_name_set(candidate_csv) if candidate_csv.is_file() else set()
    known_labels = load_known_labels_from_wheel() if cfg.normalize_via_wheel else None

    media_roots = None
    if cfg.require_media:
        media_roots = MediaRoots(
            audio=root / "audio",
            video=root / "video",
            face=root / "openface_face",
        )

    after, summary = apply_filters(
        before,
        cfg,
        media_roots=media_roots,
        human_names=human_names,
        candidate_names=candidate_names,
        known_labels=known_labels,
    )
    write_filtered_csv(after, output_path)
    report = summarize_report(before, after, summary)
    if report_json:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(report_path)
    report["output_csv"] = str(output_path)
    return output_path, report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Filter MER-Caption+ training CSV")
    parser.add_argument("--input", default=None, help="Input track2_train_mercaptionplus.csv")
    parser.add_argument("--output", default=None, help="Output filtered CSV path")
    parser.add_argument(
        "--report",
        default=None,
        help="JSON report path (default: docs/reports/mercaptionplus_filter_report.json)",
    )
    parser.add_argument("--require-media", action="store_true")
    parser.add_argument("--keep-human-overlap", action="store_true")
    parser.add_argument("--exclude-candidate", action="store_true")
    parser.add_argument("--no-wheel-filter", action="store_true")
    args = parser.parse_args()

    root = get_project_root()
    report_path = args.report or str(root / "docs/reports/mercaptionplus_filter_report.json")
    cfg = FilterConfig(
        require_media=args.require_media,
        exclude_human_names=not args.keep_human_overlap,
        exclude_candidate_overlap=args.exclude_candidate,
        normalize_via_wheel=not args.no_wheel_filter,
    )
    out_path, report = run_filter(
        input_csv=args.input,
        output_csv=args.output,
        report_json=report_path,
        cfg=cfg,
    )
    print(f"Wrote filtered CSV: {out_path}")
    print(f"Kept {report['output_count']}/{report['input_count']} samples")
    if report.get("removed"):
        print("Removed:", report["removed"])


if __name__ == "__main__":
    main()
