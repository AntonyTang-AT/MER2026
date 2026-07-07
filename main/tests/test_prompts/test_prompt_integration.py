from pathlib import Path

import pytest

from src.inference.openset_postprocess import postprocess_openset
from src.prompts.description_builder import build_description_prompt, load_routing_map
from src.prompts.openset_builder import build_openset_prompt


def test_routing_json_to_description_prompt():
    path = Path("outputs/routing/human_routing.json")
    if not path.is_file():
        pytest.skip("routing fixture not found")

    mapping = load_routing_map(path)
    name, routing = next(iter(mapping.items()))
    msg = build_description_prompt(
        subtitle=f"subtitle for {name}",
        routing=routing,
        variant="routing",
    )
    assert routing.contradiction_type in msg
    assert "subtitle for" in msg


def test_mock_reason_to_openset_string():
    reason = "The character seems joyful yet anxious about the outcome."
    prompt = build_openset_prompt(reason, variant="ew_aware")
    assert reason in prompt

    mock_llm_output = "Output: [joyful, anxious]"
    openset = postprocess_openset(mock_llm_output)
    assert "happy" in openset or "worried" in openset or "anxious" in openset
