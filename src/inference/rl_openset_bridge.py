"""RL reason→openset 桥接 — 官方 openset + reason cue 手术式补 recall 目标。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.mertools_bridge import load_npz_predictions, parse_openset_string
from src.inference.dtrb_boost import RECALL_TARGETS
from src.inference.openset_postprocess import format_openset_list
from src.inference.recall_boost import load_reason_map
from src.routing.wheel_incongruity import _label_to_level1

# 桥接专用 cue：排除 worried→anxious / happy→joyful 等宽映射，防止 c20k 洪水
BRIDGE_CUE_MAP: dict[str, str] = {
    "anxiety": "anxious",
    "anxious": "anxious",
    "apprehensive": "anxious",
    "nervous": "nervous",
    "nervousness": "nervous",
    "tense": "nervous",
    "joyful": "joyful",
    "joyous": "joyful",
    "delighted": "joyful",
    "elated": "joyful",
    "surprise": "surprised",
    "surprised": "surprised",
    "shocked": "surprised",
    "astonished": "surprised",
}

RECALL_SYNONYMS: dict[str, frozenset[str]] = {
    "surprised": frozenset({"surprise", "surprised", "shocked", "astonished"}),
    "anxious": frozenset({"anxious", "anxiety", "apprehensive", "uneasy"}),
    "nervous": frozenset({"nervous", "nervousness", "tense", "tension"}),
    "joyful": frozenset({"joyful", "joy", "delighted", "elated", "cheerful"}),
}
NOISE_SWAP = ("concerned", "confused", "worried")


@dataclass
class BridgeConfig:
    max_labels: int = 8
    max_add: int = 1
    allow_noise_swap: bool = True
    min_labels_before_add: int = 3
    require_gap: bool = True
    mode: str = "gap_add"  # gap_add | normalize_only | worried_swap


def _recall_cues_from_reason(reason: str) -> list[str]:
    if not reason.strip():
        return []
    reason_l = reason.lower()
    hits: list[str] = []
    seen: set[str] = set()
    for cue, tgt in sorted(BRIDGE_CUE_MAP.items(), key=lambda x: -len(x[0])):
        if tgt not in RECALL_TARGETS:
            continue
        if re.search(rf"\b{re.escape(cue)}\b", reason_l) and tgt not in seen:
            hits.append(tgt)
            seen.add(tgt)
    return hits


def _norm_labels(text: str) -> list[str]:
    return [x.lower().strip() for x in parse_openset_string(text) if str(x).strip()]


def _has_recall_target(labels: list[str], tgt: str) -> bool:
    lab_set = {x.lower() for x in labels}
    if tgt in lab_set:
        return True
    for syn in RECALL_SYNONYMS.get(tgt, frozenset({tgt})):
        if syn in lab_set:
            return True
    return False


def _find_swap_index(labels: list[str], *, pool: set[str]) -> int | None:
    for noise in NOISE_SWAP:
        for i, lab in enumerate(labels):
            if lab.lower() == noise:
                return i
    pred_l1 = {_label_to_level1(lab) for lab in labels}
    for i, lab in enumerate(labels):
        l1 = _label_to_level1(lab)
        if l1 is None:
            continue
        if lab.lower() in NOISE_SWAP:
            continue
        if any(_label_to_level1(t) == l1 and t in pool for t in RECALL_TARGETS):
            return i
    return None


def _normalize_synonyms(labels: list[str], cues: list[str]) -> list[str]:
    """reason 支持时将 openset 同义词规范为标准 recall 目标。"""
    out = list(labels)
    for i, lab in enumerate(out):
        low = lab.lower()
        for tgt in cues:
            syns = RECALL_SYNONYMS.get(tgt, frozenset())
            if low in syns and low != tgt:
                out[i] = tgt
                break
    return out


def bridge_single(
    openset_str: str,
    reason: str,
    *,
    cfg: BridgeConfig,
) -> tuple[str, list[str]]:
    """对单样本桥接 reason cue → openset recall 目标。"""
    labels = _norm_labels(openset_str)
    cues = _recall_cues_from_reason(reason)
    if cfg.mode == "normalize_only":
        if not cues:
            return openset_str, []
        new_labels = _normalize_synonyms(labels, cues)
        if new_labels == labels:
            return openset_str, []
        return format_openset_list(new_labels), ["normalize"]

    if cfg.mode == "worried_swap":
        if not cues:
            return openset_str, []
        added: list[str] = []
        pool = set(cues)
        for tgt in cues:
            if tgt not in ("anxious", "nervous"):
                continue
            if _has_recall_target(labels, tgt):
                continue
            idx = next((i for i, l in enumerate(labels) if l.lower() == "worried"), None)
            if idx is not None:
                labels[idx] = tgt
                added.append(tgt)
                break
        if not added:
            return openset_str, []
        return format_openset_list(labels[: cfg.max_labels]), added

    # default gap_add
    if not cues:
        return openset_str, []

    labels = _normalize_synonyms(labels, cues)
    missing = [t for t in cues if t in RECALL_TARGETS and not _has_recall_target(labels, t)]
    if cfg.require_gap and not missing:
        return openset_str, []

    added: list[str] = []
    pool = set(cues)
    for tgt in missing[: cfg.max_add]:
        if len(labels) >= cfg.max_labels:
            if cfg.allow_noise_swap:
                idx = _find_swap_index(labels, pool=pool)
                if idx is not None:
                    labels[idx] = tgt
                    added.append(tgt)
                    continue
            break
        labels.append(tgt)
        added.append(tgt)

    labels = labels[: cfg.max_labels]
    return format_openset_list(labels), added


def bridge_rl_openset(
    openset_preds: dict[str, str],
    reason_map: dict[str, str],
    *,
    cfg: BridgeConfig | None = None,
    process_names: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """批量桥接 RL openset。"""
    cfg = cfg or BridgeConfig()
    names = process_names or sorted(openset_preds.keys())
    out: dict[str, str] = dict(openset_preds)
    added_map: dict[str, list[str]] = {}
    for name in names:
        pred = openset_preds.get(name, "")
        reason = reason_map.get(name, "")
        new_pred, added = bridge_single(pred, reason, cfg=cfg)
        if added:
            out[name] = new_pred
            added_map[name] = added
    return out, added_map


def bridge_from_npz(
    *,
    openset_npz: Path,
    reason_npz: Path,
    out_npz: Path,
    cfg: BridgeConfig | None = None,
    process_names: list[str] | None = None,
) -> dict:
    """从 npz 读写桥接结果。"""
    from src.inference.ensemble_runner import save_merged_predictions

    openset = load_npz_predictions(openset_npz)
    reasons = load_reason_map(reason_npz)
    bridged, added_map = bridge_rl_openset(
        openset, reasons, cfg=cfg, process_names=process_names
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_merged_predictions(bridged, out_npz)
    return {
        "openset_npz": str(openset_npz),
        "reason_npz": str(reason_npz),
        "out_npz": str(out_npz),
        "n_samples": len(bridged),
        "n_bridged": len(added_map),
        "added_label_counts": _count_added(added_map),
    }


def _count_added(added_map: dict[str, list[str]]) -> dict[str, int]:
    from collections import Counter

    c: Counter[str] = Counter()
    for labs in added_map.values():
        for lab in labs:
            c[lab] += 1
    return dict(c)
