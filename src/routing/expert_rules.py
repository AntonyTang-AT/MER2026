"""masking / sarcasm / hidden_emotion / intensity_mismatch 专家规则 — L4.3。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.config_loader import load_yaml
from src.routing.modality_scorer import valence_dict

CONTRADICTION_TYPES = (
    "masking",
    "sarcasm",
    "hidden_emotion",
    "intensity_mismatch",
    "consistent",
)


@dataclass(frozen=True)
class RuleResult:
    contradiction_type: str
    involved_modalities: list[str]
    rule_name: str


def _same_sign(a: float, b: float) -> bool:
    return (a >= 0 and b >= 0) or (a < 0 and b < 0)


def _cfg() -> dict[str, Any]:
    return load_yaml("routing/contradiction_rules.yaml")


def classify_from_valences(
    text_v: float,
    audio_v: float,
    frame_v: float,
    face_v: float,
    *,
    rules_cfg: dict[str, Any] | None = None,
) -> RuleResult:
    """SIT 规则：text / speech→audio / macro→frame / micro→face。"""
    cfg = rules_cfg or _cfg()
    rules = cfg.get("rules", {})
    priority = cfg.get("priority", list(CONTRADICTION_TYPES))

    def _r(name: str) -> dict[str, float]:
        block = rules.get(name, {})
        return {k: float(v) for k, v in block.items()}

    for name in priority:
        if name == "consistent":
            continue
        r = _r(name)
        if name == "masking":
            if text_v > r.get("text_v_min", 0.6) and face_v < r.get("micro_v_max", -0.4):
                return RuleResult("masking", ["text", "face"], name)
        elif name == "sarcasm":
            if text_v > r.get("text_v_min", 0.3) and (
                audio_v < r.get("speech_v_max", -0.4)
                or frame_v < r.get("macro_v_max", -0.4)
            ):
                return RuleResult("sarcasm", ["text", "audio"], name)
        elif name == "hidden_emotion":
            if abs(frame_v) < r.get("macro_v_abs_max", 0.2) and abs(face_v) > r.get(
                "micro_v_abs_min", 0.5
            ):
                return RuleResult("hidden_emotion", ["frame", "face"], name)
        elif name == "intensity_mismatch":
            diff_min = r.get("valence_diff_min", 0.6)
            require_same = r.get("require_same_sign", True)
            if require_same and _same_sign(text_v, face_v) and abs(text_v - face_v) > diff_min:
                return RuleResult(
                    "intensity_mismatch",
                    ["text", "face"],
                    name,
                )

    return RuleResult("consistent", [], "consistent")


def classify_scores(scores: dict) -> RuleResult:
    v = valence_dict(scores)
    return classify_from_valences(
        v["text"],
        v["audio"],
        v["frame"],
        v["face"],
    )
