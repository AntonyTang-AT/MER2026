"""Prompt 模板与构建 — 阶段 4 公开 API。"""

from src.prompts.description_builder import build_description_prompt, load_routing_map
from src.prompts.openset_builder import build_openset_prompt, build_openset_prompt_batch
from src.prompts.templates import (
    build_few_shot_block,
    format_template,
    load_description_config,
    load_openset_config,
    load_pipeline_config,
    load_prompt_config,
)

__all__ = [
    "build_description_prompt",
    "build_few_shot_block",
    "build_openset_prompt",
    "build_openset_prompt_batch",
    "format_template",
    "load_description_config",
    "load_openset_config",
    "load_pipeline_config",
    "load_prompt_config",
    "load_routing_map",
]
