"""Integration test with real Human sample (optional media)."""

from pathlib import Path

import pytest

from src.data.dataset_index import MER2026Index
from src.routing.run_routing import route_sample


@pytest.fixture(scope="module")
def first_human_sample():
    index = MER2026Index.from_config()
    samples = index.load_split("human", limit=1)
    if not samples:
        pytest.skip("human split unavailable")
    return samples[0]


def test_route_sample_returns_routing_result(first_human_sample):
    r = route_sample(first_human_sample)
    assert r.name == first_human_sample.name
    assert r.contradiction_type in {
        "masking",
        "sarcasm",
        "hidden_emotion",
        "intensity_mismatch",
        "consistent",
    }
    assert abs(sum(r.fusion_weights.values()) - 1.0) < 1e-5
    assert set(r.modality_scores.keys()) == {"text", "audio", "face", "frame"}


def test_route_with_media_if_exists(first_human_sample):
    if not first_human_sample.media_exists:
        pytest.skip("media files missing")
    r = route_sample(first_human_sample)
    assert r.routing_confidence > 0.0
    assert Path(first_human_sample.audio_path).is_file()
