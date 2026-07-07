"""EW-aware 开放词汇抽取 Prompt — Stage C Qwen prompt 构建。"""

from __future__ import annotations

from src.prompts.templates import build_few_shot_block, load_openset_config, load_pipeline_config


def _official_prompt(reason: str) -> str:
    """复刻官方 reason_to_openset_qwen 模板。"""
    return (
        "Please assume the role of an expert in the field of emotions. "
        "We provide clues that may be related to the emotions of the characters. "
        "Based on the provided clues, please identify the emotional states of the main character. "
        "The main character is the one with the most detailed clues. "
        "Please separate different emotional categories with commas and output only the clearly "
        "identifiable emotional categories in a list format. "
        "If none are identified, please output an empty list. "
        "Input: We cannot recognize his emotional state; Output: [] "
        "Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] "
        f"Input: {reason}; Output: "
    )


def _ew_aware_prompt(reason: str) -> str:
    """基于 YAML 配置的 EW-aware 抽标签 Prompt。"""
    cfg = load_openset_config()
    pipeline = load_pipeline_config()
    hints = ", ".join(pipeline.ew_level1_hints) if pipeline.ew_level1_hints else (
        "happy, sad, angry, fear, surprise, disgust, neutral, worried"
    )
    few_shot = build_few_shot_block(cfg.few_shot_examples)
    instruction = cfg.instruction.strip()
    return (
        f"{instruction} "
        f"Use lowercase English emotion words aligned with common categories such as: {hints}. "
        "Output only a comma-separated list inside square brackets, with no explanation. "
        f"{few_shot} "
        f"Input: {reason}; Output: "
    )


def build_openset_prompt(reason: str, *, variant: str = "ew_aware") -> str:
    """构建单条 reason -> openset 的 Qwen prompt。"""
    if variant == "official":
        return _official_prompt(reason)
    if variant in ("ew_aware", "default"):
        return _ew_aware_prompt(reason)
    raise ValueError(f"Unknown openset prompt variant: {variant}")


def build_openset_prompt_batch(
    reasons: list[str],
    *,
    variant: str = "ew_aware",
) -> list[str]:
    """批量构建 openset prompt。"""
    return [build_openset_prompt(reason, variant=variant) for reason in reasons]
