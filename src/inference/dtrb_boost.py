"""DTRB — 分歧样本 targeted recall boost（CPU 后处理，无需 GT）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.evaluation.mertools_bridge import parse_openset_string
from src.inference.expert_router import SampleFeatures, extract_features
from src.inference.openset_postprocess import PostprocessConfig, format_openset_list, normalize_labels
from src.prompts.templates import load_pipeline_config
from src.routing.wheel_incongruity import _label_to_level1

RECALL_TARGETS = ("anxious", "joyful", "nervous", "surprised")
RECALL_CUE_MAP: dict[str, str] = {
    "anxiety": "anxious",
    "anxious": "anxious",
    "apprehensive": "anxious",
    "nervous": "nervous",
    "nervousness": "nervous",
    "uneasy": "nervous",
    "tense": "nervous",
    "stressed": "nervous",
    "worried": "anxious",
    "joy": "joyful",
    "joyful": "joyful",
    "joyous": "joyful",
    "delighted": "joyful",
    "happy": "joyful",
    "excited": "joyful",
    "pleased": "joyful",
    "enthusiastic": "joyful",
    "surprise": "surprised",
    "surprised": "surprised",
    "shocked": "surprised",
    "astonished": "surprised",
    "amazed": "surprised",
}

RECALL_PRIORITY = ("anxious", "nervous", "joyful", "surprised")
TRIPLE_WIN_PRIORITY = ("joyful", "surprised", "anxious", "nervous")
NOISE_LABELS = frozenset({"worried", "concerned", "confused"})
NOISE_SWAP_PRIORITY = ("concerned", "confused", "worried")
E15_NOISE_FOR_POOL = frozenset({"worried", "concerned", "confused"})

SourceClass = Literal["triple_win", "v3_win", "rl_empty_e15_noise", "stable_divergent", "mixed"]


@dataclass
class DTRBConfig:
    divergent_jaccard: float = 0.5
    max_add: int = 2
    max_labels: int = 8
    min_pred_labels: int = 6
    recall_targets: tuple[str, ...] = RECALL_TARGETS
    noise_swap_threshold: int = 2
    boost_on_noise: bool = False
    # exp016 v2
    add_only: bool = False
    tiered_swap: bool = False
    l1_conditional_swap: bool = False
    source_conditional: bool = False
    source_weight_e14: float = 1.0
    source_weight_e15: float = 1.0
    use_conformal_trigger: bool = False
    conformal_tau: float = 1.0
    reason_guided: bool = False


def _norm_set(text: str) -> set[str]:
    return {x.lower().strip() for x in parse_openset_string(text) if str(x).strip() and x.lower() != "neutral"}


def _dtrb_postprocess_config() -> PostprocessConfig:
    """DTRB 专用 synonym：保留 recall 目标，不映射到 worried。"""
    syn = dict(load_pipeline_config().synonym_map)
    for k in RECALL_TARGETS + ("anxiety", "nervousness", "joy", "surprise", "apprehensive"):
        syn.pop(k, None)
    return PostprocessConfig(apply_synonym_map=True, synonym_map=syn)


def _finalize(labels: list[str], *, max_labels: int) -> str:
    if not labels:
        return "[]"
    cfg = _dtrb_postprocess_config()
    normalized = normalize_labels(labels, cfg=cfg)
    seen: set[str] = set()
    final: list[str] = []
    for lab in normalized:
        key = lab.lower()
        if key in seen or key == "neutral":
            continue
        seen.add(key)
        final.append(lab)
    final = final[:max_labels]
    return format_openset_list(final)


def classify_divergent_source(
    features: SampleFeatures,
    *,
    triple_labels: set[str] | None = None,
    v3_labels: set[str] | None = None,
) -> SourceClass:
    """推理时分歧源类型（不依赖 GT）。"""
    if features.rl_empty and features.e15_noise_ratio > 0.3:
        return "rl_empty_e15_noise"
    if not features.rl_empty and features.jaccard_triple_v3union >= 0.5:
        return "stable_divergent"
    if triple_labels is not None and v3_labels is not None:
        tr = triple_labels & set(RECALL_TARGETS)
        vr = v3_labels & set(RECALL_TARGETS)
        if tr - vr:
            return "triple_win"
    return "mixed"


def _pool_labels(rl: str, e14: str, e15: str, *, source_class: SourceClass = "mixed") -> set[str]:
    pool = _norm_set(rl) | _norm_set(e14) | _norm_set(e15)
    if source_class == "rl_empty_e15_noise":
        e15_set = _norm_set(e15)
        pool -= {x for x in e15_set if x in E15_NOISE_FOR_POOL}
    return pool


def _weighted_recall_in_pool(
    rl: str,
    e14: str,
    e15: str,
    *,
    cfg: DTRBConfig,
    source_class: SourceClass,
) -> list[str]:
    """按组件源加权收集 pool 中的 recall 目标。"""
    scored: dict[str, float] = {}
    for text, weight in ((e14, cfg.source_weight_e14), (rl, 1.0), (e15, cfg.source_weight_e15)):
        for lab in _norm_set(text):
            if lab in cfg.recall_targets:
                scored[lab] = max(scored.get(lab, 0.0), weight)
    if source_class == "rl_empty_e15_noise":
        for lab in list(scored):
            if lab in E15_NOISE_FOR_POOL:
                del scored[lab]
    return sorted(scored, key=lambda t: (-scored[t], RECALL_PRIORITY.index(t) if t in RECALL_PRIORITY else 99))


def _cue_hits(pool: set[str]) -> list[str]:
    hits: list[str] = []
    seen: set[str] = set()
    for lab in pool:
        key = lab.lower()
        if key in RECALL_CUE_MAP:
            tgt = RECALL_CUE_MAP[key]
            if tgt not in seen:
                hits.append(tgt)
                seen.add(tgt)
        for cue, tgt in RECALL_CUE_MAP.items():
            if cue in key and tgt not in seen:
                hits.append(tgt)
                seen.add(tgt)
    return hits


def _noise_count(labels: list[str]) -> int:
    return sum(1 for lab in labels if lab.lower() in NOISE_LABELS)


def _wheel_nonconformity(pred: set[str], pool: set[str]) -> float:
    """EW wheel L1 缺失分 — pred 缺簇但 pool 有 recall 目标。"""
    score = 0.0
    pred_l1 = {_label_to_level1(lab) for lab in pred}
    pred_l1.discard(None)
    for tgt in RECALL_TARGETS:
        if tgt not in pool or tgt in pred:
            continue
        tgt_l1 = _label_to_level1(tgt)
        if tgt_l1 is None or tgt_l1 not in pred_l1:
            score += 1.0
    return score


def _recall_from_reason(reason: str) -> list[str]:
    if not reason.strip():
        return []
    reason_l = reason.lower()
    hits: list[str] = []
    seen: set[str] = set()
    for cue, tgt in sorted(RECALL_CUE_MAP.items(), key=lambda x: -len(x[0])):
        if tgt not in RECALL_TARGETS:
            continue
        if re.search(rf"\b{re.escape(cue)}\b", reason_l) and tgt not in seen:
            hits.append(tgt)
            seen.add(tgt)
    return hits


def _l1_allows_worried_swap(labels: list[str], pool: set[str]) -> bool:
    """仅 pred scared L1 只有 worried 且 pool 有 anxious/nervous 时允许换 worried。"""
    scared_l1 = _label_to_level1("worried")
    if scared_l1 is None:
        return True
    pred_scared = [lab for lab in labels if _label_to_level1(lab) == scared_l1]
    if not pred_scared:
        return True
    if not all(lab.lower() in NOISE_LABELS for lab in pred_scared):
        return False
    pool_scared = {lab for lab in pool if _label_to_level1(lab) == scared_l1}
    return bool(pool_scared & {"anxious", "nervous"})


def _find_swap_index(labels: list[str], tgt: str, *, cfg: DTRBConfig, pool: set[str] | None = None) -> int | None:
    if cfg.l1_conditional_swap and tgt in ("anxious", "nervous"):
        if pool is not None and not _l1_allows_worried_swap(labels, pool):
            return None
        worried_idx = next((i for i, l in enumerate(labels) if l.lower() == "worried"), None)
        if worried_idx is not None:
            return worried_idx
        if pool is not None:
            return None
    order = NOISE_SWAP_PRIORITY if cfg.tiered_swap else tuple(NOISE_LABELS)
    for noise in order:
        for i, lab in enumerate(labels):
            if lab.lower() == noise:
                return i
    return None


def _sort_candidates(
    candidates: list[str],
    *,
    source_class: SourceClass,
    cfg: DTRBConfig,
) -> list[str]:
    if source_class == "triple_win":
        prio = {t: i for i, t in enumerate(TRIPLE_WIN_PRIORITY)}
    else:
        prio = {t: i for i, t in enumerate(RECALL_PRIORITY)}
    return sorted(candidates, key=lambda t: prio.get(t, 99))


def should_boost(
    features: SampleFeatures,
    cfg: DTRBConfig,
    *,
    pred_labels: list[str] | None = None,
    pool: set[str] | None = None,
    source_class: SourceClass | None = None,
) -> bool:
    """推理时可用的分歧检测（不依赖 GT）。"""
    if cfg.source_conditional and source_class == "stable_divergent":
        return False
    if cfg.use_conformal_trigger and pred_labels is not None and pool is not None:
        nc = _wheel_nonconformity(set(pred_labels), pool)
        if nc >= cfg.conformal_tau:
            return True
        if cfg.source_conditional and source_class == "rl_empty_e15_noise":
            return True
        if cfg.source_conditional and source_class == "triple_win":
            return True
        return False
    if features.jaccard_triple_v3union < cfg.divergent_jaccard:
        return True
    if features.rl_empty and features.e15_noise_ratio > 0.3:
        return True
    if cfg.boost_on_noise and pred_labels is not None:
        if _noise_count(pred_labels) >= cfg.noise_swap_threshold:
            return True
    return False


def boost_one_sample(
    pred: str,
    rl: str,
    e14: str,
    e15: str,
    *,
    cfg: DTRBConfig | None = None,
    force: bool = False,
    features: SampleFeatures | None = None,
    reasons: dict[str, str] | None = None,
    reason_name: str | None = None,
) -> tuple[str, list[str]]:
    """对单样本注入 recall 目标标签。返回 (new_pred, added_labels)。"""
    cfg = cfg or DTRBConfig()
    labels = list(_norm_set(pred))
    if len(labels) >= cfg.max_labels and not force:
        return pred, []

    source_class: SourceClass = "mixed"
    if features is not None:
        from src.inference.expert_router import _merge_one_sample

        triple = _merge_one_sample(rl, e14, e15, include_rl=True)
        v3 = _merge_one_sample(rl, e14, e15, include_rl=False)
        source_class = classify_divergent_source(features, triple_labels=triple, v3_labels=v3)
        j = len(triple & v3) / max(len(triple | v3), 1)
        if j >= cfg.divergent_jaccard and not force:
            if not (cfg.source_conditional and source_class in ("rl_empty_e15_noise", "triple_win")):
                return pred, []

    pool = _pool_labels(rl, e14, e15, source_class=source_class if cfg.source_conditional else "mixed")
    pred_set = set(labels)
    candidates: list[str] = []

    if cfg.reason_guided and reasons and reason_name:
        reason_text = reasons.get(reason_name, "")
        for tgt in _recall_from_reason(reason_text):
            if tgt not in pred_set and tgt not in candidates:
                candidates.append(tgt)

    if cfg.source_weight_e14 != 1.0 or cfg.source_weight_e15 != 1.0 or cfg.source_conditional:
        for tgt in _weighted_recall_in_pool(rl, e14, e15, cfg=cfg, source_class=source_class):
            if tgt not in pred_set and tgt not in candidates:
                candidates.append(tgt)
    else:
        for tgt in cfg.recall_targets:
            if tgt in pool and tgt not in pred_set:
                candidates.append(tgt)

    for tgt in _cue_hits(pool):
        if tgt not in pred_set and tgt not in candidates:
            candidates.append(tgt)
    for tgt in cfg.recall_targets:
        if tgt in pred_set or tgt in candidates:
            continue
        tgt_l1 = _label_to_level1(tgt)
        if tgt_l1 is None:
            continue
        for lab in pool:
            if _label_to_level1(lab) == tgt_l1 and lab not in pred_set:
                candidates.append(tgt)
                break

    if not candidates:
        return pred, []

    candidates = _sort_candidates(candidates, source_class=source_class, cfg=cfg)

    merged = list(labels)
    added: list[str] = []
    for tgt in candidates[: cfg.max_add]:
        if tgt in {x.lower() for x in merged}:
            continue
        swap_idx = _find_swap_index(merged, tgt, cfg=cfg, pool=pool)
        if swap_idx is not None:
            merged[swap_idx] = tgt
            added.append(tgt)
        elif not cfg.add_only or len(merged) < cfg.max_labels:
            if len(merged) < cfg.max_labels:
                merged.append(tgt)
                added.append(tgt)

    if not added:
        return pred, []
    return _finalize(merged, max_labels=cfg.max_labels), added


def apply_dtrb_boost(
    preds: dict[str, str],
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    *,
    cfg: DTRBConfig | None = None,
    names: list[str] | None = None,
    reasons: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """批量 DTRB boost。返回 (preds, name->added)。"""
    cfg = cfg or DTRBConfig()
    names = names or sorted(set(preds) & set(rl) & set(e14) & set(e15))
    out: dict[str, str] = dict(preds)
    added_map: dict[str, list[str]] = {}

    for name in names:
        feats = extract_features(name, rl, e14, e15)
        pred_labels = list(_norm_set(preds.get(name, "")))
        from src.inference.expert_router import _merge_one_sample

        triple = _merge_one_sample(rl.get(name, ""), e14.get(name, ""), e15.get(name, ""), include_rl=True)
        v3 = _merge_one_sample(rl.get(name, ""), e14.get(name, ""), e15.get(name, ""), include_rl=False)
        source_class = classify_divergent_source(feats, triple_labels=triple, v3_labels=v3)
        pool = _pool_labels(
            rl.get(name, ""), e14.get(name, ""), e15.get(name, ""),
            source_class=source_class if cfg.source_conditional else "mixed",
        )
        if not should_boost(
            feats, cfg, pred_labels=pred_labels, pool=pool, source_class=source_class,
        ):
            continue
        new_pred, added = boost_one_sample(
            preds.get(name, "[]"),
            rl.get(name, "[]"),
            e14.get(name, "[]"),
            e15.get(name, "[]"),
            cfg=cfg,
            features=feats,
            reasons=reasons,
            reason_name=name,
        )
        if added:
            out[name] = new_pred
            added_map[name] = added
    return out, added_map


def calibrate_conformal_tau(
    preds: dict[str, str],
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    *,
    divergent_names: list[str],
    quantile: float = 0.8,
) -> float:
    """val306 分歧子集上校准 conformal 分位数 τ。"""
    scores: list[float] = []
    for name in divergent_names:
        pred_labels = _norm_set(preds.get(name, ""))
        pool = _pool_labels(rl.get(name, ""), e14.get(name, ""), e15.get(name, ""))
        scores.append(_wheel_nonconformity(pred_labels, pool))
    if not scores:
        return 1.0
    scores.sort()
    idx = min(int(len(scores) * quantile), len(scores) - 1)
    return max(scores[idx], 0.5)
