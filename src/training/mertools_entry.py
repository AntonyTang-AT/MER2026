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


def apply_torch_amp_compat() -> None:
    """MERTools 仅白名单 torch 2.1/2.4/2.6；2.8+cu128（5090）走相同 amp API。"""
    import torch

    raw = torch.__version__
    if not (raw.startswith("2.8.") or raw.startswith("2.11.")):
        return

    class _TorchVersion(str):
        def startswith(self, prefix: str, *args, **kwargs) -> bool:  # type: ignore[override]
            if prefix in ("2.4.0", "2.6.0") and super().startswith(("2.8.", "2.11.")):
                return True
            return super().startswith(prefix, *args, **kwargs)

    torch.__version__ = _TorchVersion(raw)  # type: ignore[misc]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.mertools_entry <script.py> [args...]")
        sys.exit(1)

    apply_compat_shims()
    apply_torch_amp_compat()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
