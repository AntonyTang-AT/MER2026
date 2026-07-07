from src.prompts.openset_builder import build_openset_prompt, build_openset_prompt_batch


def test_official_prompt_contains_reason():
    reason = "He looks angry but says he is fine."
    prompt = build_openset_prompt(reason, variant="official")
    assert reason in prompt
    assert "expert" in prompt.lower()


def test_ew_aware_prompt_contains_reason_and_few_shot():
    reason = "She appears joyful and worried."
    prompt = build_openset_prompt(reason, variant="ew_aware")
    assert reason in prompt
    assert "Input:" in prompt
    assert "happy" in prompt.lower()


def test_batch_matches_single():
    reasons = ["a", "b"]
    batch = build_openset_prompt_batch(reasons, variant="ew_aware")
    assert len(batch) == 2
    assert batch[0] == build_openset_prompt("a", variant="ew_aware")
