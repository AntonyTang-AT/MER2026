"""openset 后处理 — Qwen 原始输出 → 评估友好字符串。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.config_loader import get_project_root, load_global_config
from src.evaluation.mertools_bridge import parse_openset_string
from src.prompts.templates import load_pipeline_config


@dataclass
class PostprocessConfig:
    lowercase: bool = True
    deduplicate: bool = True
    apply_synonym_map: bool = True
    filter_unknown: bool = False
    synonym_map: dict[str, str] | None = None
    known_labels: set[str] | None = None


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
    """从 pipeline.yaml 构建默认后处理配置。"""
    pipeline = load_pipeline_config()
    pp = pipeline.postprocess
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


def normalize_labels(
    labels: list[str],
    *,
    cfg: PostprocessConfig,
) -> list[str]:
    """规范化标签列表：小写、同义词、去重、可选过滤。"""
    out: list[str] = []
    seen: set[str] = set()
    synonym_map = cfg.synonym_map or {}

    for label in labels:
        word = label.strip()
        if cfg.lowercase:
            word = word.lower()
        if not word:
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


def format_openset_list(labels: list[str]) -> str:
    """格式化为官方 npz fileitems 字符串。"""
    if not labels:
        return "[]"
    inner = ", ".join(labels)
    return f"[{inner}]"


def postprocess_openset(raw: str, *, cfg: PostprocessConfig | None = None) -> str:
    """完整后处理管道：清洗 → 解析 → 规范化 → 格式化。"""
    cfg = cfg or default_postprocess_config(mode="ew")
    cleaned = strip_qwen_prefix(raw)
    labels = parse_openset_string(cleaned)
    normalized = normalize_labels(labels, cfg=cfg)
    return format_openset_list(normalized)
