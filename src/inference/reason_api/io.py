"""Reason npz 读写（与 AffectGPT / recall_boost 对齐）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_reason_map(path: Path | str) -> dict[str, str]:
    """读取 reason npz → name->reason。"""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "name2reason" in data:
        raw = data["name2reason"]
        # AffectGPT: 0-d object with .item(); 部分脚本写过 .tolist()
        try:
            mapping = raw.item()
        except Exception:
            mapping = raw.tolist()
        if isinstance(mapping, dict):
            return {str(k): str(v) for k, v in mapping.items()}
    if "filenames" in data and "fileitems" in data:
        return {
            str(n): str(t)
            for n, t in zip(data["filenames"], data["fileitems"])
        }
    raise ValueError(f"Unsupported reason npz keys={list(data.keys())}: {path}")


def save_reason_map(name2reason: dict[str, str], path: Path | str) -> Path:
    """写出 AffectGPT 兼容的 name2reason npz。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 保持 dict 于 object array，供 .item() / 部分 .tolist() 路径使用
    payload = {str(k): str(v) for k, v in name2reason.items()}
    np.savez_compressed(path, name2reason=payload)
    return path


def load_names_json(path: Path | str) -> list[str]:
    """读取分歧子集 JSON：samples[].name / names / 纯 list。"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "name" in data[0]:
            return [str(x["name"]) for x in data]
        return [str(x) for x in data]
    if isinstance(data, dict):
        if "names" in data:
            return [str(x) for x in data["names"]]
        if "samples" in data:
            return [str(x["name"]) for x in data["samples"] if "name" in x]
    raise ValueError(f"Cannot parse names from {path}")
