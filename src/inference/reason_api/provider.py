"""Reason 外部 API Provider：可插拔接入 OpenAI 兼容大模型。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from src.inference.reason_api.types import ReasonAPIConfig, ReasonSample
from src.prompts.reason_api_builder import (
    build_generate_from_clues_prompt,
    build_refine_prompt,
)


def resolve_api_key(cfg: ReasonAPIConfig) -> str:
    """按配置与常见环境变量解析 API Key。"""
    candidates = [
        cfg.api_key_env,
        "REASON_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]
    seen: set[str] = set()
    for env_name in candidates:
        if not env_name or env_name in seen:
            continue
        seen.add(env_name)
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    return ""


class ReasonAPIProvider(ABC):
    """外部 Reason API 抽象接口。

    典型接入点（在 AffectGPT reason 之后、openset / RRB 之前）：
      AffectGPT reason_npz → provider.refine_map(...) → refined_reason.npz → openset_extractor / RRB
    """

    def __init__(self, cfg: ReasonAPIConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def refine_one(self, sample: ReasonSample) -> str:
        """润色/改写已有 reason 文本。"""

    def generate_from_clues(self, sample: ReasonSample) -> str:
        """仅文本线索生成 reason（默认未实现，子类可覆盖）。"""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_from_clues; "
            "use mode=refine or implement this method."
        )

    def process_one(self, sample: ReasonSample) -> str:
        mode = (self.cfg.mode or "refine").strip().lower()
        if mode == "refine":
            return self.refine_one(sample)
        if mode in ("generate_from_clues", "generate"):
            return self.generate_from_clues(sample)
        raise ValueError(f"Unknown reason API mode: {self.cfg.mode!r}")

    def process_many(self, samples: Iterable[ReasonSample]) -> dict[str, str]:
        items = list(samples)
        if not items:
            return {}
        if self.cfg.dry_run:
            return {s.name: (s.reason or s.clues or "") for s in items}

        workers = max(1, int(self.cfg.max_concurrency))
        out: dict[str, str] = {}
        if workers == 1:
            for s in items:
                out[s.name] = self._safe_process(s)
            return out

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self._safe_process, s): s.name for s in items}
            for fut in as_completed(futs):
                name = futs[fut]
                out[name] = fut.result()
        return out

    def refine_map(self, name2reason: dict[str, str], *, names: list[str] | None = None) -> dict[str, str]:
        """便捷：dict[str,str] → 改写后的 dict（未选中的 name 原样保留）。"""
        selected = names if names is not None else list(name2reason.keys())
        samples = [
            ReasonSample(name=n, reason=name2reason.get(n, ""))
            for n in selected
            if n in name2reason
        ]
        refined = self.process_many(samples)
        out = dict(name2reason)
        out.update(refined)
        return out

    def _safe_process(self, sample: ReasonSample) -> str:
        try:
            text = self.process_one(sample).strip()
            if text:
                return text
            if self.cfg.fallback_to_input:
                return (sample.reason or sample.clues or "").strip()
            return text
        except Exception:
            if self.cfg.fallback_to_input:
                return (sample.reason or sample.clues or "").strip()
            raise


class PassthroughReasonAPI(ReasonAPIProvider):
    """不调用外部 API，原样返回（联调 / dry-run）。"""

    def refine_one(self, sample: ReasonSample) -> str:
        return (sample.reason or "").strip()

    def generate_from_clues(self, sample: ReasonSample) -> str:
        return (sample.clues or sample.reason or "").strip()


class OpenAICompatibleReasonAPI(ReasonAPIProvider):
    """OpenAI Chat Completions 兼容接口（DeepSeek / OpenAI / 中转 / vLLM）。"""

    def __init__(self, cfg: ReasonAPIConfig) -> None:
        super().__init__(cfg)
        self._client = None

    def __enter__(self) -> OpenAICompatibleReasonAPI:
        if self.cfg.dry_run:
            return self
        api_key = resolve_api_key(self.cfg)
        if not api_key:
            raise RuntimeError(
                "Reason API key missing. Set REASON_API_KEY (or OPENAI_API_KEY / DEEPSEEK_API_KEY)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package required: pip install openai") from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout_s,
        )
        return self

    def __exit__(self, *args: object) -> None:
        self._client = None

    def _chat(self, user_prompt: str) -> str:
        if self.cfg.dry_run:
            return user_prompt[:200]
        if self._client is None:
            raise RuntimeError("OpenAICompatibleReasonAPI not entered; use `with provider:`")
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        choice = resp.choices[0].message
        return str(getattr(choice, "content", "") or "").strip()

    def refine_one(self, sample: ReasonSample) -> str:
        prompt = build_refine_prompt(sample, variant=self.cfg.prompt_variant)
        return self._chat(prompt)

    def generate_from_clues(self, sample: ReasonSample) -> str:
        variant = "generate_v1"
        if self.cfg.prompt_variant.startswith("generate"):
            variant = self.cfg.prompt_variant
        prompt = build_generate_from_clues_prompt(sample, variant=variant)
        return self._chat(prompt)


class _NullContext:
    def __init__(self, provider: ReasonAPIProvider) -> None:
        self.provider = provider

    def __enter__(self) -> ReasonAPIProvider:
        return self.provider

    def __exit__(self, *args: object) -> None:
        return None


def build_provider(cfg: ReasonAPIConfig) -> ReasonAPIProvider:
    name = (cfg.provider or "openai_compatible").strip().lower()
    if name in ("passthrough", "identity", "none"):
        return PassthroughReasonAPI(cfg)
    if name in ("openai_compatible", "openai", "deepseek", "vllm"):
        return OpenAICompatibleReasonAPI(cfg)
    raise ValueError(
        f"Unknown reason API provider: {cfg.provider!r}. "
        "Supported: openai_compatible, passthrough"
    )


def open_provider(cfg: ReasonAPIConfig) -> _NullContext | OpenAICompatibleReasonAPI:
    """统一 `with open_provider(cfg) as api:` 用法。"""
    provider = build_provider(cfg)
    if isinstance(provider, OpenAICompatibleReasonAPI):
        return provider
    return _NullContext(provider)
