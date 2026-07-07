"""MERTools 运行时入口 — 应用兼容补丁后执行官方脚本。"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def apply_compat_shims() -> None:
    """pytorchvideo 0.1.5 依赖已移除的 torchvision.transforms.functional_tensor。"""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.mertools_entry <script.py> [args...]")
        sys.exit(1)

    apply_compat_shims()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
