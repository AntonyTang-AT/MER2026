"""Pipeline 数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingResult:
    name: str
    contradiction_type: str = "consistent"
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {"text": 0.25, "audio": 0.25, "face": 0.25, "frame": 0.25}
    )
    routing_confidence: float = 1.0
    modality_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class OpensetPrediction:
    name: str
    openset: list[str]
    raw_text: str = ""


@dataclass
class SampleContext:
    """单样本全链路状态。"""

    name: str
    subtitle: str = ""
    routing: RoutingResult | None = None
    description: str = ""
    openset: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
