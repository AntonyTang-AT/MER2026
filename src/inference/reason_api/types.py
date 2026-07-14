"""Reason 外部 API：类型与配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasonSample:
    """单条 reason 改写/生成请求。"""

    name: str
    reason: str = ""
    subtitle: str = ""
    clues: str = ""  # 可选：额外文本线索（日志、OCR 等）
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasonAPIConfig:
    """外部 reason API 运行配置（可用 YAML / 环境变量覆盖）。"""

    provider: str = "openai_compatible"
    # OpenAI 兼容端点（DeepSeek / OpenAI / 各类中转 / 本地 vLLM）
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key_env: str = "REASON_API_KEY"  # 依次回退 OPENAI_API_KEY、DEEPSEEK_API_KEY
    temperature: float = 0.3
    max_tokens: int = 768
    timeout_s: float = 120.0
    max_concurrency: int = 4
    # refine | generate_from_clues
    mode: str = "refine"
    prompt_variant: str = "refine_v1"
    system_prompt: str = (
        "You are an expert multimodal emotion analyst. "
        "Rewrite emotional clues clearly and factually in English. "
        "Do not invent events absent from the input. "
        "Output only the rewritten clue paragraph, no preamble."
    )
    # 失败时是否回退到原始 reason
    fallback_to_input: bool = True
    dry_run: bool = False
