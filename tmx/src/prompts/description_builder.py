"""权重注入描述 Prompt — Stage B user_message 构建。"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.types import RoutingResult
from src.prompts.templates import format_template, load_description_config, load_pipeline_config

OFFICIAL_OVLABEL_QUESTION = "Please recognize all possible emotional states of the character."

_MODALITY_FOCUS_HINTS = {
    "text": "Pay special attention to subtitle wording and semantic cues.",
    "audio": "Pay special attention to vocal tone, pitch, and acoustic patterns.",
    "face": "Pay special attention to facial micro-expressions and fine-grained face dynamics.",
    "frame": "Pay special attention to body language and scene-level visual context.",
}


def _official_question() -> str:
    cfg = load_pipeline_config()
    return cfg.official.get("description_question", OFFICIAL_OVLABEL_QUESTION)


def _focus_hint(weights: dict[str, float], threshold: float = 0.4) -> str:
    if not weights:
        return ""
    top_modality = max(weights, key=weights.get)  # type: ignore[arg-type]
    if weights[top_modality] >= threshold:
        return _MODALITY_FOCUS_HINTS.get(top_modality, "")
    return ""


def build_description_prompt(
    *,
    subtitle: str,
    routing: RoutingResult | None = None,
    variant: str = "default",
) -> str:
    """构建 AffectGPT inference 的 user_message。"""
    if variant == "official":
        return _official_question()

    desc_cfg = load_description_config()

    if variant == "routing" and routing is not None:
        weights = routing.fusion_weights
        body = format_template(
            desc_cfg.template_with_routing,
            w_text=f"{weights.get('text', 0.25):.2f}",
            w_audio=f"{weights.get('audio', 0.25):.2f}",
            w_face=f"{weights.get('face', 0.25):.2f}",
            w_frame=f"{weights.get('frame', 0.25):.2f}",
            contradiction_type=routing.contradiction_type,
            subtitle=subtitle,
        )
        hint = _focus_hint(weights)
        if hint:
            body = f"{body}\n{hint}"
        return body

    return format_template(desc_cfg.template_default, subtitle=subtitle)


def load_routing_map(path: Path | str) -> dict[str, RoutingResult]:
    """从 routing JSON 加载 name -> RoutingResult 映射。"""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, RoutingResult] = {}
    for item in payload:
        result[item["name"]] = RoutingResult(
            name=item["name"],
            contradiction_type=item.get("contradiction_type", "consistent"),
            fusion_weights=dict(item.get("fusion_weights") or {}),
            routing_confidence=float(item.get("routing_confidence", 1.0)),
            modality_scores=dict(item.get("modality_scores") or {}),
        )
    return result
