"""Emotion Wheel 空间不一致分数 — 替代纯 Jaccard/VA 阈值。"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from src.core.config_loader import get_project_root, load_global_config
from src.core.types import ModalitySelfOutput
from src.training.data_filter import load_known_labels_from_wheel

LEVEL1_ORDER = ("mad", "disgusted", "sad", "happy", "surprised", "scared")


@lru_cache(maxsize=1)
def _wheel_level1_index() -> dict[str, int]:
    """label(lower) -> level1 cluster index。"""
    cfg = load_global_config()
    wheel_path = (
        get_project_root() / cfg["paths"]["mertools_root"] / "emotion_wheel" / "wheel_mapping.npz"
    )
    if not wheel_path.is_file():
        return {}
    data = np.load(wheel_path, allow_pickle=True)
    format_mapping = data["format_mapping"].item()
    wheel_map = data["wheel_map_whole"].item()
    level1_map = wheel_map.get("wheel1", {}).get("level1", {})

    label_to_l1: dict[str, int] = {}
    l1_to_idx = {name: i for i, name in enumerate(LEVEL1_ORDER)}

    for raw, clusters in format_mapping.items():
        key = str(raw).strip().lower()
        if not clusters:
            continue
        canon = str(clusters[0]).strip().lower()
        l1 = level1_map.get(canon, canon)
        if l1 in l1_to_idx:
            label_to_l1[key] = l1_to_idx[l1]
        for part in key.replace("-", " ").split():
            if part in label_to_l1:
                continue
            if part in level1_map:
                label_to_l1[part] = l1_to_idx[level1_map[part]]

    for canon, l1 in level1_map.items():
        if l1 in l1_to_idx:
            label_to_l1[str(canon).lower()] = l1_to_idx[l1]
    return label_to_l1


def _label_to_level1(label: str) -> int | None:
    key = label.strip().lower()
    idx_map = _wheel_level1_index()
    if key in idx_map:
        return idx_map[key]
    for token in key.replace("-", " ").split():
        if token in idx_map:
            return idx_map[token]
    for known, idx in idx_map.items():
        if known in key or key in known:
            return idx
    return None


def openset_to_wheel_vector(openset: list[str]) -> list[float]:
    """openset -> level1 分布向量（L1 归一化）。"""
    n = len(LEVEL1_ORDER)
    counts = [0.0] * n
    for lab in openset:
        idx = _label_to_level1(lab)
        if idx is not None:
            counts[idx] += 1.0
    total = sum(counts)
    if total <= 0:
        return [1.0 / n] * n
    return [c / total for c in counts]


def vector_distance(a: list[float], b: list[float]) -> float:
    """L1 距离，范围约 [0, 2]。"""
    return sum(abs(x - y) for x, y in zip(a, b))


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def modality_wheel_vector(per_modality: dict[str, ModalitySelfOutput]) -> dict[str, list[float]]:
    return {
        mod: openset_to_wheel_vector(out.openset)
        for mod, out in per_modality.items()
        if out.openset
    }


def pairwise_wheel_distance(
    per_modality: dict[str, ModalitySelfOutput],
) -> dict[str, float]:
    vecs = modality_wheel_vector(per_modality)
    mods = list(vecs.keys())
    out: dict[str, float] = {}
    for i, a in enumerate(mods):
        for b in mods[i + 1 :]:
            d = vector_distance(vecs[a], vecs[b])
            out[f"{a}|{b}"] = min(1.0, d / 2.0)
    return out


def max_incongruity(per_modality: dict[str, ModalitySelfOutput]) -> float:
    pairwise = pairwise_wheel_distance(per_modality)
    if not pairwise:
        return 0.0
    return max(pairwise.values())


def wheel_confidence_boost(per_modality: dict[str, ModalitySelfOutput]) -> float:
    """高 wheel 不一致 → 更高结构识别置信度。"""
    return max_incongruity(per_modality)
