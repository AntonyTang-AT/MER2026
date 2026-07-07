"""矛盾类型 → fusion_weights 查表 — L4.4。"""

from __future__ import annotations

from typing import Any

from src.core.config_loader import load_yaml

DEFAULT_WEIGHTS = {
    "text": 0.25,
    "audio": 0.25,
    "face": 0.25,
    "frame": 0.25,
}


def _load_table() -> dict[str, dict[str, float]]:
    raw = load_yaml("routing/weight_table.yaml")
    table: dict[str, dict[str, float]] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        weights = {k: float(v) for k, v in val.items() if k in DEFAULT_WEIGHTS}
        if weights:
            table[key] = weights
    return table


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.get(k, 0.0) for k in DEFAULT_WEIGHTS)
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: weights.get(k, 0.0) / total for k in DEFAULT_WEIGHTS}


def routing_confidence(
    contradiction_type: str,
    intensity: float,
    *,
    max_intensity: float = 1.2,
) -> float:
    if contradiction_type == "consistent":
        return max(0.5, 1.0 - intensity / max_intensity)
    return min(1.0, max(0.3, intensity / max_intensity))


def select_weights(
    contradiction_type: str,
    intensity: float = 0.0,
    *,
    table: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, float], float]:
    tbl = table or _load_table()
    raw = tbl.get(contradiction_type, tbl.get("consistent", DEFAULT_WEIGHTS))
    weights = normalize_weights(raw)
    conf = routing_confidence(contradiction_type, intensity)
    return weights, conf
