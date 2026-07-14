"""RL ∪ e14 ∪ e15 三路 union merge — exp014 triple 生产默认。"""

from __future__ import annotations

from typing import Literal

from src.evaluation.mertools_bridge import parse_openset_string
from src.inference.openset_postprocess import (
    Tier2PostprocessConfig,
    format_openset_list,
    postprocess_labels_tier2,
    sanitize_openset_string,
)

SourceKey = Literal["rl", "e14", "e15"]
MergeMode = Literal["fifo", "priority", "e15_deprioritize"]
WheelCapMode = Literal["wcc_l1_1", "wcc_l1_2", "wcc_l1_1_deprioritize"]

DEFAULT_SOURCE_ORDER: tuple[SourceKey, ...] = ("rl", "e14", "e15")
E15_DEPRIORITIZE_LABELS = frozenset({"confused", "concerned", "worried"})


def _norm_label(
    lab: str,
    *,
    replace_neutral: str | None,
) -> tuple[str, str] | None:
    key = lab.lower()
    if key == "neutral":
        if replace_neutral:
            lab = replace_neutral
            key = lab.lower()
        else:
            return None
    return lab, key


def _collect_labeled(
    sources: dict[SourceKey, dict[str, str]],
    name: str,
    *,
    source_order: tuple[SourceKey, ...],
    replace_neutral: str | None,
    deprioritize_labels: frozenset[str],
    deprioritize_source: SourceKey | None,
) -> list[tuple[str, str, SourceKey]]:
    """返回 (label, key, source) 列表，供排序与 cap 截断。"""
    items: list[tuple[str, str, SourceKey]] = []
    seen: set[str] = set()
    for src_key in source_order:
        raw = sources.get(src_key, {})
        for lab in parse_openset_string(raw.get(name, "")):
            normed = _norm_label(lab, replace_neutral=replace_neutral)
            if normed is None:
                continue
            lab, key = normed
            if key in seen:
                continue
            seen.add(key)
            items.append((lab, key, src_key))
    if deprioritize_labels:
        normal: list[tuple[str, str, SourceKey]] = []
        low: list[tuple[str, str, SourceKey]] = []
        for lab, key, sk in items:
            if deprioritize_source and sk == deprioritize_source and key in deprioritize_labels:
                low.append((lab, key, sk))
            elif key in deprioritize_labels:
                low.append((lab, key, sk))
            else:
                normal.append((lab, key, sk))
        items = normal + low
    return items


def _finalize_labels(labels: list[str], *, max_labels: int) -> str:
    labels = labels[:max_labels]
    if labels:
        sanitized = parse_openset_string(
            sanitize_openset_string("[" + ", ".join(labels) + "]")
        )
    else:
        sanitized = []
    seen2: set[str] = set()
    final: list[str] = []
    for lab in sanitized:
        key = lab.lower()
        if key == "neutral" or key in seen2:
            continue
        seen2.add(key)
        final.append(lab)
    final = final[:max_labels]
    return format_openset_list(final)


def merge_triple_union(
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    *,
    max_labels: int = 8,
    replace_neutral: str | None = None,
    source_order: tuple[SourceKey, ...] = DEFAULT_SOURCE_ORDER,
    mode: MergeMode = "fifo",
    deprioritize_labels: frozenset[str] | None = None,
) -> dict[str, str]:
    """三路 union + sanitize；mode 控制标签优先级策略。"""
    sources: dict[SourceKey, dict[str, str]] = {"rl": rl, "e14": e14, "e15": e15}
    dep_src: SourceKey | None = None
    dep_set = deprioritize_labels or frozenset()
    if mode == "e15_deprioritize":
        dep_set = deprioritize_labels or E15_DEPRIORITIZE_LABELS
        dep_src = "e15"
    elif mode == "priority":
        pass  # source_order 已表达 RL > e14 > e15
    # fifo: 默认 source_order 顺序

    out: dict[str, str] = {}
    all_names = sorted(set(rl) | set(e14) | set(e15))
    for name in all_names:
        items = _collect_labeled(
            sources,
            name,
            source_order=source_order,
            replace_neutral=replace_neutral,
            deprioritize_labels=dep_set,
            deprioritize_source=dep_src,
        )
        labels = [lab for lab, _, _ in items]
        out[name] = _finalize_labels(labels, max_labels=max_labels)
    return out


def merge_triple_from_npz_paths(
    rl_path: str,
    e14_path: str,
    e15_path: str,
    **kwargs,
) -> dict[str, str]:
    from pathlib import Path

    from src.evaluation.mertools_bridge import load_npz_predictions

    rl = load_npz_predictions(Path(rl_path))
    e14 = load_npz_predictions(Path(e14_path))
    e15 = load_npz_predictions(Path(e15_path))
    return merge_triple_union(rl, e14, e15, **kwargs)


def apply_wheel_cap(
    preds: dict[str, str],
    *,
    max_per_l1: int = 1,
    max_labels: int = 8,
) -> dict[str, str]:
    """对 merge 结果施加 EW L1 分桶 cap（WCC 后处理）。"""
    cfg = Tier2PostprocessConfig(
        l1_dedup=True,
        max_per_l1=max_per_l1,
        max_labels=max_labels,
        adaptive_cap=False,
    )
    out: dict[str, str] = {}
    for name, text in preds.items():
        labels = parse_openset_string(text)
        final = postprocess_labels_tier2(labels, cfg=cfg)
        out[name] = format_openset_list(final)
    return out


def merge_triple_with_wheel_cap(
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    *,
    wheel_mode: WheelCapMode = "wcc_l1_1",
    max_labels: int = 8,
    replace_neutral: str | None = None,
) -> dict[str, str]:
    """三路 union + WCC 变体。"""
    if wheel_mode == "wcc_l1_1":
        merged = merge_triple_union(rl, e14, e15, max_labels=max_labels, mode="fifo", replace_neutral=replace_neutral)
        return apply_wheel_cap(merged, max_per_l1=1, max_labels=max_labels)
    if wheel_mode == "wcc_l1_2":
        merged = merge_triple_union(rl, e14, e15, max_labels=max_labels, mode="fifo", replace_neutral=replace_neutral)
        return apply_wheel_cap(merged, max_per_l1=2, max_labels=max_labels)
    if wheel_mode == "wcc_l1_1_deprioritize":
        merged = merge_triple_union(
            rl, e14, e15, max_labels=max_labels, mode="e15_deprioritize", replace_neutral=replace_neutral
        )
        return apply_wheel_cap(merged, max_per_l1=1, max_labels=max_labels)
    raise ValueError(f"Unknown wheel_mode: {wheel_mode!r}")
