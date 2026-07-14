"""Reason 外部 API 的 prompt 构建。"""

from __future__ import annotations

from src.inference.reason_api.types import ReasonSample


def build_refine_prompt(sample: ReasonSample, *, variant: str = "refine_v1") -> str:
    """将已有 AffectGPT reason 交给外部 LLM 润色/补全。"""
    reason = (sample.reason or "").strip()
    extra = []
    if sample.subtitle.strip():
        extra.append(f"Subtitle: {sample.subtitle.strip()}")
    if sample.clues.strip():
        extra.append(f"Additional clues: {sample.clues.strip()}")
    extras = ("\n".join(extra) + "\n") if extra else ""

    if variant == "refine_v1":
        return (
            "Rewrite the following character-emotion clues into a clearer English paragraph "
            "suitable for open-vocabulary emotion label extraction.\n"
            "Keep all supported emotional evidence; you may clarify high-arousal states "
            "(anxious, joyful, nervous, surprised) when the clues support them.\n"
            "Do not add unsupported plots. Output ONLY the rewritten paragraph.\n\n"
            f"{extras}"
            f"Original clues:\n{reason}\n\n"
            "Rewritten clues:"
        )
    if variant == "refine_compact":
        return (
            "Compress and clarify the emotion clues below in one short English paragraph. "
            "Preserve emotion-related facts only.\n\n"
            f"{extras}"
            f"{reason}"
        )
    raise ValueError(f"Unknown refine prompt variant: {variant}")


def build_generate_from_clues_prompt(sample: ReasonSample, *, variant: str = "generate_v1") -> str:
    """无视频时：仅凭文本线索生成 reason（占位能力，需调用方提供 clues/subtitle）。"""
    parts = []
    if sample.subtitle.strip():
        parts.append(f"Subtitle: {sample.subtitle.strip()}")
    if sample.clues.strip():
        parts.append(f"Clues: {sample.clues.strip()}")
    if sample.reason.strip():
        parts.append(f"Draft: {sample.reason.strip()}")
    body = "\n".join(parts) if parts else "(no textual clues provided)"

    if variant == "generate_v1":
        return (
            "Based only on the textual information below, write an English paragraph describing "
            "the main character's likely emotional states and supporting cues.\n"
            "Do not invent visual events that are not implied. Output ONLY the paragraph.\n\n"
            f"{body}\n\n"
            "Emotion clue paragraph:"
        )
    raise ValueError(f"Unknown generate prompt variant: {variant}")
