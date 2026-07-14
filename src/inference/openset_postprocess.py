"""openset 后处理 — Qwen 原始输出 → 评估友好字符串。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.config_loader import get_project_root, load_global_config, load_yaml
from src.evaluation.mertools_bridge import parse_openset_string
from src.prompts.templates import load_pipeline_config
from src.routing.wheel_incongruity import LEVEL1_ORDER, _label_to_level1


@dataclass
class PostprocessConfig:
    lowercase: bool = True
    deduplicate: bool = True
    apply_synonym_map: bool = True
    filter_unknown: bool = False
    synonym_map: dict[str, str] | None = None
    known_labels: set[str] | None = None


@dataclass
class Tier2PostprocessConfig(PostprocessConfig):
    l1_dedup: bool = False
    max_per_l1: int = 2
    max_labels: int = 8
    adaptive_cap: bool = False


def strip_qwen_prefix(raw: str) -> str:
    """移植官方 func_postprocess_qwen 前缀清洗逻辑。"""
    response = raw.strip()
    prefixes = (
        "输入",
        "输出",
        "翻译",
        "让我们来翻译一下：",
        "output",
        "Output",
        "input",
        "Input",
    )
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix) :]
    response = response.strip()
    if response.startswith(":"):
        response = response[1:]
    if response.startswith("："):
        response = response[1:]
    response = response.strip().replace("\n", "")
    return response.strip()


def _load_known_labels_from_wheel() -> set[str]:
    """从 emotion_wheel/wheel_mapping.npz 读取 format_mapping 键集合。"""
    cfg = load_global_config()
    wheel_path = (
        get_project_root()
        / cfg["paths"]["mertools_root"]
        / "emotion_wheel"
        / "wheel_mapping.npz"
    )
    if not wheel_path.is_file():
        return set()
    data = np.load(wheel_path, allow_pickle=True)
    format_mapping = data["format_mapping"].tolist()
    return {str(k).lower().strip() for k in format_mapping.keys()}


def default_postprocess_config(*, mode: str = "ew") -> PostprocessConfig:
    """从 pipeline.yaml 构建默认后处理配置。

    mode:
      - ew: 生产默认，全程 synonym（anxious/nervous→worried 等）
      - raw: 仅小写+去重，不做 synonym（提交面实验）
      - dtrb_preserve / rl_recall: 保留 recall 细标的 synonym 子集
    """
    if mode in ("rl_recall", "dtrb_preserve"):
        from src.inference.dtrb_boost import _dtrb_postprocess_config

        return _dtrb_postprocess_config()
    pipeline = load_pipeline_config()
    pp = pipeline.postprocess
    if mode == "raw":
        return PostprocessConfig(
            lowercase=bool(pp.get("lowercase", True)),
            deduplicate=bool(pp.get("deduplicate", True)),
            apply_synonym_map=False,
            filter_unknown=False,
            synonym_map={},
            known_labels=None,
        )
    synonym_map = pipeline.synonym_map if mode == "ew" else {}
    known = _load_known_labels_from_wheel() if pp.get("filter_unknown") else None
    return PostprocessConfig(
        lowercase=bool(pp.get("lowercase", True)),
        deduplicate=bool(pp.get("deduplicate", True)),
        apply_synonym_map=bool(pp.get("apply_synonym_map", mode == "ew")),
        filter_unknown=bool(pp.get("filter_unknown", False)),
        synonym_map=synonym_map,
        known_labels=known,
    )


def tier2_config_from_yaml() -> Tier2PostprocessConfig:
    """从 pipeline.yaml tier2.postprocess 构建配置。"""
    block = load_yaml("pipeline.yaml").get("tier2", {}).get("postprocess", {})
    base = default_postprocess_config(mode="ew")
    filter_unknown = bool(block.get("filter_unknown", base.filter_unknown))
    known = _load_known_labels_from_wheel() if filter_unknown else base.known_labels
    return Tier2PostprocessConfig(
        lowercase=base.lowercase,
        deduplicate=base.deduplicate,
        apply_synonym_map=base.apply_synonym_map,
        filter_unknown=filter_unknown,
        synonym_map=base.synonym_map,
        known_labels=known,
        l1_dedup=bool(block.get("l1_dedup", True)),
        max_per_l1=int(block.get("max_per_l1", 2)),
        max_labels=int(block.get("max_labels", 8)),
        adaptive_cap=bool(block.get("adaptive_cap", False)),
    )


def _l1_name(idx: int | None) -> str:
    if idx is None or idx < 0 or idx >= len(LEVEL1_ORDER):
        return "unknown"
    return LEVEL1_ORDER[idx]


def dedup_by_l1(
    labels: list[str],
    *,
    max_per_l1: int = 2,
) -> list[str]:
    """每个 EW L1 簇最多保留 max_per_l1 个标签。"""
    if max_per_l1 <= 0:
        return labels
    buckets: dict[str, list[str]] = {}
    unknown: list[str] = []
    for lab in labels:
        idx = _label_to_level1(lab)
        key = _l1_name(idx)
        if key == "unknown":
            unknown.append(lab)
            continue
        buckets.setdefault(key, []).append(lab)
    out: list[str] = []
    for key in sorted(buckets.keys()):
        out.extend(buckets[key][:max_per_l1])
    out.extend(unknown[:max_per_l1])
    return out


def adaptive_cap(
    labels: list[str],
    *,
    base_max: int = 8,
    confidence: float = 1.0,
) -> list[str]:
    """低置信样本减少标签上限，抑制标签洪水。"""
    conf = max(0.0, min(1.0, confidence))
    if conf >= 0.7:
        cap = base_max
    elif conf >= 0.4:
        cap = max(4, base_max - 2)
    else:
        cap = max(3, base_max - 4)
    return labels[:cap]


def normalize_labels(
    labels: list[str],
    *,
    cfg: PostprocessConfig,
) -> list[str]:
    """规范化标签列表：小写、同义词、去重、可选过滤。"""
    out: list[str] = []
    seen: set[str] = set()
    synonym_map = cfg.synonym_map or {}
    invalid = {"]", "[", ",", ""}

    for label in labels:
        word = label.strip()
        if cfg.lowercase:
            word = word.lower()
        if not word or word in invalid:
            continue
        if cfg.apply_synonym_map:
            word = synonym_map.get(word, word)
        if cfg.filter_unknown and cfg.known_labels is not None:
            if word not in cfg.known_labels:
                continue
        if cfg.deduplicate:
            if word in seen:
                continue
            seen.add(word)
        out.append(word)
    return out


def postprocess_labels_tier2(
    labels: list[str],
    *,
    cfg: Tier2PostprocessConfig,
    confidence: float = 1.0,
) -> list[str]:
    """Tier2 标签级后处理管道。"""
    normalized = normalize_labels(labels, cfg=cfg)
    if cfg.l1_dedup:
        normalized = dedup_by_l1(normalized, max_per_l1=cfg.max_per_l1)
    if cfg.adaptive_cap:
        normalized = adaptive_cap(normalized, base_max=cfg.max_labels, confidence=confidence)
    elif len(normalized) > cfg.max_labels:
        normalized = normalized[: cfg.max_labels]
    return normalized


def format_openset_list(labels: list[str]) -> str:
    """格式化为官方 npz fileitems 字符串。"""
    clean = [str(x).strip() for x in labels if str(x).strip() not in ("", "]", "[", ",")]
    if not clean:
        return "[]"
    inner = ", ".join(clean)
    return f"[{inner}]"


def sanitize_openset_string(value: str, *, mode: str = "ew") -> str:
    """提交前清洗 openset：去空标签、 stray bracket token。

    mode 见 default_postprocess_config；默认 ew 保持历史生产行为。
    """
    labels = parse_openset_string(value)
    cfg = default_postprocess_config(mode=mode)
    labels = normalize_labels(labels, cfg=cfg)
    return format_openset_list(labels)


def postprocess_tier2(
    raw_or_labels: str | list[str],
    *,
    cfg: Tier2PostprocessConfig | None = None,
    confidence: float = 1.0,
) -> str:
    """Tier2 完整后处理：解析 → 规范化 → L1 dedup → cap → 格式化。"""
    cfg = cfg or tier2_config_from_yaml()
    if isinstance(raw_or_labels, str):
        cleaned = strip_qwen_prefix(raw_or_labels)
        labels = parse_openset_string(cleaned)
    else:
        labels = list(raw_or_labels)
    final = postprocess_labels_tier2(labels, cfg=cfg, confidence=confidence)
    return format_openset_list(final)


def postprocess_openset(raw: str, *, cfg: PostprocessConfig | None = None) -> str:
    """完整后处理管道：清洗 → 解析 → 规范化 → 格式化。"""
    cfg = cfg or default_postprocess_config(mode="ew")
    cleaned = strip_qwen_prefix(raw)
    labels = parse_openset_string(cleaned)
    normalized = normalize_labels(labels, cfg=cfg)
    return format_openset_list(normalized)
