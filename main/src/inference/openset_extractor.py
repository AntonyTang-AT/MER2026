"""Qwen 开放词汇标签抽取 — 封装官方 ovlabel_extraction.py。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.core.config_loader import load_yaml
from src.training.mertools_paths import get_paths, sync_mertools_config


def run_ovlabel_extraction(
    *,
    reason_npz: Path | None = None,
    store_npz: Path | None = None,
    cuda_devices: str | None = None,
    process_all: bool = False,
) -> int:
    """调用官方 ovlabel_extraction 逻辑。

    process_all=True 时直接运行官方脚本（扫描 output/results-mer2026ov）。
    否则对单个 reason npz 调用 extract_openset_batchcalling。
    """
    sync_mertools_config(verbose=False)
    paths = get_paths()
    mertools_root = paths["mertools_root"]
    project_root = paths["project_root"]
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    if process_all:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.training.mertools_entry",
            "ovlabel_extraction.py",
        ]
        print("Running:", " ".join(cmd), f"(cwd={mertools_root})")
        return subprocess.call(cmd, cwd=str(mertools_root), env=env)

    if reason_npz is None or store_npz is None:
        raise ValueError("reason_npz and store_npz required when process_all=False")

    bl = load_yaml("baseline.yaml")
    modelname = bl["ovlabel"]["modelname"]

    code = f"""
import runpy, sys
sys.path.insert(0, {str(project_root)!r})
from src.training.mertools_entry import apply_compat_shims
apply_compat_shims()
sys.path.insert(0, {str(mertools_root)!r})
import ovlabel_extraction as ov
ov.extract_openset_batchcalling(
    reason_npz={str(reason_npz)!r},
    store_npz={str(store_npz)!r},
    modelname={modelname!r},
)
"""
    print(f"Extract openset: {reason_npz} -> {store_npz}")
    return subprocess.call([sys.executable, "-c", code], env=env)


def openset_path_for_reason(reason_npz: Path) -> Path:
    name = reason_npz.name
    if name.endswith(".npz"):
        return reason_npz.with_name(name[:-4] + "-openset.npz")
    return reason_npz.parent / f"{reason_npz.stem}-openset.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract openset labels via Qwen (ovlabel)")
    parser.add_argument("--reason-npz", type=Path, default=None, help="Single reason npz")
    parser.add_argument("--store-npz", type=Path, default=None, help="Output openset npz")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="process_all",
        help="Run official batch ovlabel_extraction.py",
    )
    parser.add_argument("--cuda", default=None)
    args = parser.parse_args()

    store = args.store_npz
    if args.reason_npz and store is None:
        store = openset_path_for_reason(args.reason_npz)

    code = run_ovlabel_extraction(
        reason_npz=args.reason_npz,
        store_npz=store,
        cuda_devices=args.cuda,
        process_all=args.process_all,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
