"""YAML 模板加载与占位符填充。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.config_loader import load_yaml


@dataclass
class DescriptionPromptConfig:
    system_role: str = ""
    template_with_routing: str = ""
    template_default: str = ""


@dataclass
class OpensetPromptConfig:
    instruction: str = ""
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    postprocess: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelinePromptConfig:
    default_variant: dict[str, str] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    synonym_map: dict[str, str] = field(default_factory=dict)
    ew_level1_hints: list[str] = field(default_factory=list)
    postprocess: dict[str, Any] = field(default_factory=dict)
    official: dict[str, str] = field(default_factory=dict)


def format_template(template: str, **kwargs: Any) -> str:
    """安全填充模板占位符，缺失键时抛出 KeyError。"""
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing template placeholder: {missing}") from exc


def build_few_shot_block(
    examples: list[dict[str, str]],
    *,
    style: str = "input_output",
) -> str:
    """将 few-shot 示例转为 Qwen 兼容的 Input/Output 块。"""
    if style != "input_output":
        raise ValueError(f"Unsupported few-shot style: {style}")

    lines: list[str] = []
    for ex in examples:
        inp = ex.get("input", "").strip()
        out = ex.get("output", "").strip()
        lines.append(f"Input: {inp}; Output: {out}")
    return " \\\n".join(lines)


def load_description_config() -> DescriptionPromptConfig:
    raw = load_yaml("prompts/description.yaml")
    return DescriptionPromptConfig(
        system_role=str(raw.get("system_role", "")).strip(),
        template_with_routing=str(raw.get("template_with_routing", "")).strip(),
        template_default=str(raw.get("template_default", "")).strip(),
    )


def load_openset_config() -> OpensetPromptConfig:
    raw = load_yaml("prompts/openset_extract.yaml")
    examples = raw.get("few_shot_examples") or []
    return OpensetPromptConfig(
        instruction=str(raw.get("instruction", "")).strip(),
        few_shot_examples=[dict(item) for item in examples],
        postprocess=dict(raw.get("postprocess") or {}),
    )


def load_pipeline_config() -> PipelinePromptConfig:
    raw = load_yaml("prompts/pipeline.yaml")
    return PipelinePromptConfig(
        default_variant=dict(raw.get("default_variant") or {}),
        paths=dict(raw.get("paths") or {}),
        synonym_map={str(k).lower(): str(v).lower() for k, v in (raw.get("synonym_map") or {}).items()},
        ew_level1_hints=[str(x) for x in (raw.get("ew_level1_hints") or [])],
        postprocess=dict(raw.get("postprocess") or {}),
        official=dict(raw.get("official") or {}),
    )


def load_prompt_config(name: str) -> DescriptionPromptConfig | OpensetPromptConfig | PipelinePromptConfig:
    """按名称加载 prompt 配置。"""
    if name == "description":
        return load_description_config()
    if name == "openset":
        return load_openset_config()
    if name == "pipeline":
        return load_pipeline_config()
    raise ValueError(f"Unknown prompt config name: {name}")
