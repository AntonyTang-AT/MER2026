from src.inference.openset_postprocess import (
    PostprocessConfig,
    format_openset_list,
    normalize_labels,
    postprocess_openset,
    strip_qwen_prefix,
)


def test_strip_qwen_prefix():
    assert strip_qwen_prefix("Output: [happy, sad]") == "[happy, sad]"
    assert strip_qwen_prefix("输入：happy") == "happy"


def test_postprocess_empty():
    cfg = PostprocessConfig(synonym_map={"joyful": "happy"})
    assert postprocess_openset("[]", cfg=cfg) == "[]"
    assert postprocess_openset("", cfg=cfg) == "[]"


def test_postprocess_synonym_dedupe():
    cfg = PostprocessConfig(
        lowercase=True,
        deduplicate=True,
        apply_synonym_map=True,
        synonym_map={"joyful": "happy", "joy": "happy"},
    )
    raw = "Output: [joyful, happy, Joy]"
    out = postprocess_openset(raw, cfg=cfg)
    assert out == "[happy]"


def test_normalize_labels_filter_unknown():
    cfg = PostprocessConfig(
        lowercase=True,
        deduplicate=True,
        apply_synonym_map=False,
        filter_unknown=True,
        known_labels={"happy", "sad"},
    )
    labels = normalize_labels(["happy", "unknown_word", "sad"], cfg=cfg)
    assert labels == ["happy", "sad"]


def test_format_openset_list():
    assert format_openset_list(["a", "b"]) == "[a, b]"
