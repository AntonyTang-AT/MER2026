"""Human-OV train/val 划分与标签统计。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.core.config_loader import get_project_root, load_yaml
from src.data.dataset_index import MER2026Index


def _split_cfg() -> dict:
    return load_yaml("dataset.yaml")["splits"]


def _human_csv_path() -> Path:
    idx = MER2026Index.from_config()
    return idx._csv_path("human")


def load_human_names() -> list[str]:
    path = _human_csv_path()
    df = pd.read_csv(path)
    return [str(x) for x in df["name"].tolist()]


def make_split() -> tuple[list[str], list[str]]:
    cfg = _split_cfg()
    names = load_human_names()
    ratio = float(cfg["human_ov_val_ratio"])
    seed = int(cfg["human_ov_val_seed"])

    n_val = int(len(names) * ratio)
    shuffled = pd.Series(names).sample(frac=1.0, random_state=seed).tolist()
    val_names = shuffled[:n_val]
    train_names = shuffled[n_val:]
    return train_names, val_names


def val_split_path() -> Path:
    cfg = _split_cfg()
    rel = cfg["human_ov_val_list"]
    return (get_project_root() / rel).resolve()


def write_val_split(path: Path | None = None) -> Path:
    _, val_names = make_split()
    out = path or val_split_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(val_names) + "\n", encoding="utf-8")
    return out


def load_val_names() -> list[str]:
    path = val_split_path()
    if not path.is_file():
        write_val_split(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_train_names() -> list[str]:
    val_set = set(load_val_names())
    return [n for n in load_human_names() if n not in val_set]


def generate_data_stats(out_path: Path | None = None) -> str:
    idx = MER2026Index.from_config()
    samples = idx.load_split("human")
    val_names = set(load_val_names())
    train_names = [s.name for s in samples if s.name not in val_names]

    label_counts: Counter[str] = Counter()
    per_sample_counts: list[int] = []
    single_label = 0
    multi_label = 0

    for sample in samples:
        n_labels = len(sample.openset)
        per_sample_counts.append(n_labels)
        if n_labels <= 1:
            single_label += 1
        else:
            multi_label += 1
        for label in sample.openset:
            label_counts[label] += 1

    unique_labels = len(label_counts)
    avg_labels = sum(per_sample_counts) / len(per_sample_counts) if per_sample_counts else 0.0
    top30 = label_counts.most_common(30)

    lines = [
        "# MER2026 Human-OV 标签统计",
        "",
        f"- 总样本数: {len(samples)}",
        f"- 训练集: {len(train_names)}",
        f"- 验证集: {len(val_names)}",
        f"- 唯一标签数: {unique_labels}",
        f"- 每样本标签数: min={min(per_sample_counts)}, max={max(per_sample_counts)}, mean={avg_labels:.2f}",
        f"- 单标签样本: {single_label} ({100 * single_label / len(samples):.1f}%)",
        f"- 多标签样本: {multi_label} ({100 * multi_label / len(samples):.1f}%)",
        "",
        "## Top-30 高频标签",
        "",
        "| 标签 | 出现次数 |",
        "|------|----------|",
    ]
    for label, count in top30:
        lines.append(f"| {label} | {count} |")

    content = "\n".join(lines) + "\n"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-OV train/val split and stats")
    parser.add_argument("--write", action="store_true", help="Write human_ov_val.txt")
    parser.add_argument("--stats", action="store_true", help="Generate data stats markdown")
    parser.add_argument(
        "--out",
        type=Path,
        default=get_project_root() / "docs/reports/data_stats.md",
        help="Output path for stats markdown",
    )
    args = parser.parse_args()

    if args.write:
        path = write_val_split()
        train, val = make_split()
        print(f"Wrote val split: {path} ({len(val)} samples, train={len(train)})")

    if args.stats:
        out = generate_data_stats(args.out)
        if args.out:
            print(f"Wrote stats: {args.out}")
        else:
            print(out)

    if not args.write and not args.stats:
        train, val = make_split()
        print(f"Human-OV: total={len(train) + len(val)}, train={len(train)}, val={len(val)}")


if __name__ == "__main__":
    main()
