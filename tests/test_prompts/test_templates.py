import pytest

from src.prompts.templates import (
    build_few_shot_block,
    format_template,
    load_description_config,
    load_openset_config,
    load_pipeline_config,
)


def test_load_description_config():
    cfg = load_description_config()
    assert "expert" in cfg.template_default.lower()
    assert "{subtitle}" in cfg.template_with_routing or "subtitle" in cfg.template_with_routing


def test_load_openset_config_has_few_shot():
    cfg = load_openset_config()
    assert len(cfg.few_shot_examples) >= 2
    assert cfg.few_shot_examples[0]["input"]


def test_load_pipeline_config_synonym_map():
    cfg = load_pipeline_config()
    assert cfg.synonym_map.get("joyful") == "happy"
    assert "routing_json" in cfg.paths


def test_format_template_success():
    out = format_template("Hello {name}", name="world")
    assert out == "Hello world"


def test_format_template_missing_key():
    with pytest.raises(KeyError, match="subtitle"):
        format_template("Sub: {subtitle}")


def test_build_few_shot_block():
    examples = [
        {"input": "a", "output": "[]"},
        {"input": "b", "output": "[happy]"},
    ]
    block = build_few_shot_block(examples)
    assert "Input: a; Output: []" in block
    assert "Input: b; Output: [happy]" in block
