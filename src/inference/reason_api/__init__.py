"""Reason 外部 API 接入层。

用法概要::

    from src.inference.reason_api import (
        ReasonAPIConfig,
        build_provider,
        load_reason_map,
        save_reason_map,
    )

    cfg = ReasonAPIConfig(base_url="https://api.deepseek.com", model="deepseek-chat")
    with build_provider(cfg) as api:  # Passthrough 无 __enter__ 时需特殊处理
        ...

推荐用 scripts/run_reason_api_refine.py 批量改写 reason npz，再送入 openset / RRB。
"""

from __future__ import annotations

from src.inference.reason_api.io import load_names_json, load_reason_map, save_reason_map
from src.inference.reason_api.provider import (
    OpenAICompatibleReasonAPI,
    PassthroughReasonAPI,
    ReasonAPIProvider,
    build_provider,
    open_provider,
    resolve_api_key,
)
from src.inference.reason_api.types import ReasonAPIConfig, ReasonSample

__all__ = [
    "ReasonAPIConfig",
    "ReasonSample",
    "ReasonAPIProvider",
    "OpenAICompatibleReasonAPI",
    "PassthroughReasonAPI",
    "build_provider",
    "open_provider",
    "resolve_api_key",
    "load_reason_map",
    "save_reason_map",
    "load_names_json",
    "load_reason_api_config",
]


def load_reason_api_config(path: str | None = None) -> ReasonAPIConfig:
    """从 config/reason_api.yaml 加载配置（可用环境变量覆盖关键字段）。"""
    import os
    from pathlib import Path

    import yaml

    from src.core.config_loader import get_project_root, load_yaml
    from src.inference.reason_api.types import ReasonAPIConfig

    if path is None:
        raw = load_yaml("reason_api.yaml") or {}
    else:
        yaml_path = Path(path)
        if not yaml_path.is_file():
            yaml_path = get_project_root() / path
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    cfg = ReasonAPIConfig(
        provider=str(raw.get("provider", "openai_compatible")),
        base_url=str(raw.get("base_url", "https://api.deepseek.com")),
        model=str(raw.get("model", "deepseek-chat")),
        api_key_env=str(raw.get("api_key_env", "REASON_API_KEY")),
        temperature=float(raw.get("temperature", 0.3)),
        max_tokens=int(raw.get("max_tokens", 768)),
        timeout_s=float(raw.get("timeout_s", 120.0)),
        max_concurrency=int(raw.get("max_concurrency", 4)),
        mode=str(raw.get("mode", "refine")),
        prompt_variant=str(raw.get("prompt_variant", "refine_v1")),
        system_prompt=str(
            raw.get(
                "system_prompt",
                ReasonAPIConfig().system_prompt,
            )
        ),
        fallback_to_input=bool(raw.get("fallback_to_input", True)),
        dry_run=bool(raw.get("dry_run", False)),
    )

    # 环境变量覆盖，方便临时换模型
    if os.environ.get("REASON_API_BASE_URL", "").strip():
        cfg.base_url = os.environ["REASON_API_BASE_URL"].strip()
    if os.environ.get("REASON_API_MODEL", "").strip():
        cfg.model = os.environ["REASON_API_MODEL"].strip()
    if os.environ.get("REASON_API_PROVIDER", "").strip():
        cfg.provider = os.environ["REASON_API_PROVIDER"].strip()
    if os.environ.get("REASON_API_MODE", "").strip():
        cfg.mode = os.environ["REASON_API_MODE"].strip()
    if os.environ.get("REASON_API_DRY_RUN", "").strip() in ("1", "true", "True"):
        cfg.dry_run = True
    return cfg
