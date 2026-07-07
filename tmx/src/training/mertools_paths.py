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


def sync_train_configs(*, verbose: bool = True) -> list[str]:
    """将 config/train/*.yaml 同步到 MERTools train_configs/。"""
    paths = get_paths()
    src_dir = paths["project_root"] / "config" / "train"
    dst_dir = paths["mertools_root"] / "train_configs"
    copied: list[str] = []
    if not src_dir.is_dir():
        return copied
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.glob("*.yaml")):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)
        if verbose:
            print(f"synced train config: {src.name}")
    return copied


def patch_mercaptionplus_label_csv(label_csv: Path | str, *, verbose: bool = True) -> bool:
    """将 config.PATH_TO_LABEL['MERCaptionPlus'] 指向过滤后 CSV。"""
    paths = get_paths()
    config_py = paths["mertools_root"] / "config.py"
    label_str = str(Path(label_csv).resolve()).replace("\\", "/")
    text = config_py.read_text(encoding="utf-8")
    patched, n = re.subn(
        r"(\s*'MERCaptionPlus':\s*)[^\n]+",
        rf"\1os.path.join(DATA_DIR['MER2026'], '{Path(label_csv).name}'),",
        text,
        count=1,
    )
    if n == 0:
        raise RuntimeError("Could not patch PATH_TO_LABEL['MERCaptionPlus'] in config.py")
    if patched != text:
        config_py.write_text(patched, encoding="utf-8")
        if verbose:
            print(f"patched MERCaptionPlus label csv: {label_str}")
        return True
    return False


def restore_mercaptionplus_label_csv(*, verbose: bool = True) -> bool:
    """恢复 MERCaptionPlus 标签路径为官方默认 CSV。"""
    return patch_mercaptionplus_label_csv("track2_train_mercaptionplus.csv", verbose=verbose)


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
    train_cfgs = sync_train_configs(verbose=verbose)

    result = {
        "mertools_root": str(mertools_root),
        "data_root": str(data_root),
        "models_root": str(models_root),
        "models_link": str(mertools_root / "models"),
        "config_patched": str(changed),
        "train_configs_synced": ",".join(train_cfgs) if train_cfgs else "",
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
