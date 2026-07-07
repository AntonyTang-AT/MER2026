"""YAML 配置加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"


def get_project_root() -> Path:
    return _PROJECT_ROOT


def load_yaml(name: str) -> dict[str, Any]:
    """加载 config/ 下 yaml，name 可含子路径如 routing/weight_table。"""
    path = _CONFIG_DIR / name
    if not path.suffix:
        path = path.with_suffix(".yaml")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_global_config() -> dict[str, Any]:
    return load_yaml("global.yaml")
