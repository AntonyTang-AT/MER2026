from pathlib import Path

import pytest

from src.core.types import RoutingResult
from src.prompts.description_builder import (
    OFFICIAL_OVLABEL_QUESTION,
    build_description_prompt,
    load_routing_map,
)


@pytest.fixture
def sample_routing() -> RoutingResult:
    return RoutingResult(
        name="test_sample",
        contradiction_type="masking",
        fusion_weights={"text": 0.1, "audio": 0.2, "face": 0.5, "frame": 0.2},
        routing_confidence=0.9,
    )


def test_official_variant():
    msg = build_description_prompt(subtitle="Hello", variant="official")
    assert msg == OFFICIAL_OVLABEL_QUESTION


def test_default_variant_includes_subtitle():
    msg = build_description_prompt(subtitle="I am happy today", variant="default")
    assert "I am happy today" in msg


def test_routing_variant_includes_weights(sample_routing):
    msg = build_description_prompt(
        subtitle="Fine.",
        routing=sample_routing,
        variant="routing",
    )
    assert "masking" in msg
    assert "0.50" in msg or "0.5" in msg
    assert "Fine." in msg
    assert "micro-expressions" in msg.lower() or "facial" in msg.lower()


def test_load_routing_map_from_fixture():
    path = Path("outputs/routing/human_routing.json")
    if not path.is_file():
        pytest.skip("routing fixture not found")
    mapping = load_routing_map(path)
    assert len(mapping) > 0
    first = next(iter(mapping.values()))
    assert first.contradiction_type
    assert sum(first.fusion_weights.values()) == pytest.approx(1.0, abs=0.01)
