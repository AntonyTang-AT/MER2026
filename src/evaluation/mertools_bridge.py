"""MERTools 官方评估代码桥接。"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.core.config_loader import get_project_root, load_global_config


def get_mertools_root() -> Path:
    cfg = load_global_config()
    rel = cfg["paths"]["mertools_root"]
    root = (get_project_root() / rel).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MERTools root not found: {root}")
    return root


@contextmanager
def mertools_context() -> Iterator[Path]:
    """将 cwd 与 sys.path 临时切换到 MER2026_Track2，供官方 import config 使用。"""
    mertools_root = get_mertools_root()
    prev_cwd = os.getcwd()
    inserted = str(mertools_root)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
        path_added = True
    else:
        path_added = False

    os.chdir(mertools_root)
    try:
        yield mertools_root
    finally:
        os.chdir(prev_cwd)
        if path_added:
            sys.path.remove(inserted)


def parse_openset_string(value: object) -> list[str]:
    """与官方 toolkit.utils.functions.string_to_list 行为一致。"""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value)
    if text == "":
        return []
    if text[0] == "[":
        text = text[1:]
    if text and text[-1] == "]":
        text = text[:-1]
    return [item.strip() for item in re.split(r"['\",]", text) if item.strip() not in ("", ",")]


def import_wheel() -> tuple[Any, Any]:
    """返回 (wheel_metric_calculation, parse_openset_string)。

    通过 importlib 直接加载 wheel.py，避免触发 my_affectgpt 包 __init__ 的重依赖。
    """
    with mertools_context() as mertools_root:
        wheel_path = mertools_root / "my_affectgpt/evaluation/wheel.py"
        spec = importlib.util.spec_from_file_location("_mertools_wheel_eval", wheel_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load wheel module: {wheel_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.wheel_metric_calculation, parse_openset_string


def load_npz_predictions(path: Path | str) -> dict[str, str]:
    """读取官方 openset npz：filenames + fileitems。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction npz not found: {path}")

    data = np.load(path, allow_pickle=True)
    if "filenames" not in data or "fileitems" not in data:
        raise ValueError(f"Invalid npz (need filenames/fileitems): {path}")

    names = data["filenames"]
    items = data["fileitems"]
    return {str(name): str(item) for name, item in zip(names, items)}


def load_gt_from_csv(csv_path: Path | str, openset_col: str = "openset") -> dict[str, str]:
    """从 CSV 读取 name -> openset 字符串。"""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "name" not in df.columns or openset_col not in df.columns:
        raise ValueError(f"CSV must have 'name' and '{openset_col}' columns: {csv_path}")

    name2gt: dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row["name"])
        openset = row[openset_col]
        if pd.isna(openset):
            openset = ""
        else:
            openset = str(openset)
        name2gt[name] = openset
    return name2gt
