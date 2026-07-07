"""多 checkpoint openset 融合 — 阶段 6.4。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.mertools_bridge import load_npz_predictions, parse_openset_string
from src.inference.openset_postprocess import PostprocessConfig, format_openset_list, normalize_labels


def merge_openset_predictions(
    name2pred_list: list[dict[str, str]],
    *,
    strategy: str = "label_union",
    min_votes: int = 2,
    postprocess_cfg: PostprocessConfig | None = None,
) -> dict[str, str]:
    """合并多个 name->openset 字符串预测。"""
    if not name2pred_list:
        return {}

    all_names: set[str] = set()
    for pred in name2pred_list:
        all_names.update(pred.keys())

    merged: dict[str, str] = {}
    cfg = postprocess_cfg or PostprocessConfig(
        lowercase=True,
        deduplicate=True,
        apply_synonym_map=False,
    )

    for name in sorted(all_names):
        label_lists: list[list[str]] = []
        for pred in name2pred_list:
            raw = pred.get(name, "[]")
            label_lists.append(parse_openset_string(raw))

        if strategy == "label_union":
            combined: list[str] = []
            seen: set[str] = set()
            for labels in label_lists:
                for label in labels:
                    word = label.strip().lower()
                    if word and word not in seen:
                        seen.add(word)
                        combined.append(word)
            normalized = normalize_labels(combined, cfg=cfg)
            merged[name] = format_openset_list(normalized)
            continue

        if strategy == "majority_vote":
            counts: dict[str, int] = {}
            for labels in label_lists:
                for label in labels:
                    word = label.strip().lower()
                    if not word:
                        continue
                    counts[word] = counts.get(word, 0) + 1
            winners = sorted(
                word for word, count in counts.items() if count >= min_votes
            )
            normalized = normalize_labels(winners, cfg=cfg)
            merged[name] = format_openset_list(normalized)
            continue

        raise ValueError(f"Unknown ensemble strategy: {strategy}")

    return merged


def merge_openset_npz(
    npz_paths: list[Path | str],
    *,
    strategy: str = "label_union",
    min_votes: int = 2,
) -> dict[str, str]:
    """从多个 openset npz 融合预测。"""
    preds = [load_npz_predictions(Path(p)) for p in npz_paths]
    return merge_openset_predictions(preds, strategy=strategy, min_votes=min_votes)


def ensemble_and_save(
    npz_paths: list[Path | str],
    out_npz: Path | str,
    *,
    strategy: str = "label_union",
) -> Path:
    """融合多个 openset npz 并写入新 npz。"""
    out_npz = Path(out_npz)
    merged = merge_openset_npz(npz_paths, strategy=strategy)
    names = list(merged.keys())
    items = [merged[name] for name in names]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, filenames=names, fileitems=items)
    return out_npz
