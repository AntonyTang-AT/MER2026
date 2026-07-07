"""MERTools 路径同步 — 对齐官方 config.py 与项目 global.yaml。"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from src.core.config_loader import get_project_root, load_global_config, load_yaml


def get_paths() -> dict[str, Path]:
    global_cfg = load_global_config()
    root = get_project_root()
    return {
        "project_root": root,
        "mertools_root": (root / global_cfg["paths"]["mertools_root"]).resolve(),
        "data_root": (root / global_cfg["paths"]["data_root"]).resolve(),
        "models_root": (root / global_cfg["paths"]["models_root"]).resolve(),
        "outputs_root": (root / global_cfg["paths"]["outputs_root"]).resolve(),
    }


def _ensure_models_symlink(mertools_root: Path, models_root: Path) -> None:
    link = mertools_root / "models"
    if link.is_symlink():
        if link.resolve() == models_root.resolve():
            return
        link.unlink()
    elif link.is_dir():
        return
    elif link.exists():
        link.unlink()
    link.symlink_to(models_root, target_is_directory=True)


def _patch_config_py(mertools_root: Path, data_root: Path) -> bool:
    config_py = mertools_root / "config.py"
    text = config_py.read_text(encoding="utf-8")
    data_str = str(data_root).replace("\\", "/")

    new_line = f"    'MER2026':          '{data_str}',"
    patched, n = re.subn(
        r"(\s*'MER2026':\s*')[^']*(')",
        rf"\1{data_str}\2",
        text,
        count=1,
    )
    if n == 0:
        raise RuntimeError("Could not patch DATA_DIR['MER2026'] in config.py")

    if patched != text:
        config_py.write_text(patched, encoding="utf-8")
        return True
    return False


def sync_mertools_config(*, verbose: bool = True) -> dict[str, str]:
    paths = get_paths()
    mertools_root = paths["mertools_root"]
    data_root = paths["data_root"]
    models_root = paths["models_root"]

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root missing: {data_root}")
    if not models_root.is_dir():
        raise FileNotFoundError(f"Models root missing: {models_root}")

    _ensure_models_symlink(mertools_root, models_root)
    changed = _patch_config_py(mertools_root, data_root)

    result = {
        "mertools_root": str(mertools_root),
        "data_root": str(data_root),
        "models_root": str(models_root),
        "models_link": str(mertools_root / "models"),
        "config_patched": str(changed),
    }
    if verbose:
        for k, v in result.items():
            print(f"{k}: {v}")
    return result


def verify_mertools_config() -> list[str]:
    """返回缺失项列表，空表示通过。"""
    paths = get_paths()
    issues: list[str] = []

    mertools_root = paths["mertools_root"]
    data_root = paths["data_root"]
    models_root = paths["models_root"]

    required_models = [
        "Qwen2.5-7B-Instruct/config.json",
        "clip-vit-large-patch14/config.json",
        "chinese-hubert-large/config.json",
    ]
    for rel in required_models:
        if not (models_root / rel).is_file():
            issues.append(f"missing model: {rel}")

    for sub in ("audio", "video", "openface_face"):
        d = data_root / sub
        if not d.is_dir():
            issues.append(f"missing data dir: {sub}/")
        elif not any(d.iterdir()):
            issues.append(f"empty data dir: {sub}/")

    config_py = mertools_root / "config.py"
    if config_py.is_file():
        text = config_py.read_text(encoding="utf-8")
        if "xxx/dataset" in text:
            issues.append("config.py DATA_DIR still has placeholder xxx")

    wheel = mertools_root / "emotion_wheel/wheel_mapping.npz"
    if not wheel.is_file():
        issues.append("missing emotion_wheel/wheel_mapping.npz (run scripts/sync_emotion_wheel.sh)")

    return issues
