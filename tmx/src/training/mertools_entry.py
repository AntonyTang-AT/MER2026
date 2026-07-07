"""MERTools 运行时入口 — 应用兼容补丁后执行官方脚本。"""

from __future__ import annotations

import ast
import importlib.util
import os
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

    try:
        import transformers.modeling_utils as modeling_utils
        from transformers import pytorch_utils

        for name in (
            "apply_chunking_to_forward",
            "find_pruneable_heads_and_indices",
            "prune_linear_layer",
        ):
            if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
                setattr(modeling_utils, name, getattr(pytorch_utils, name))
    except Exception:
        pass


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


def _load_script_module(script_path: Path) -> object:
    name = script_path.stem
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _exec_script_main(script_path: Path, module_globals: dict) -> None:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.comparators) != 1:
            continue
        comp = test.comparators[0]
        if isinstance(comp, ast.Constant) and comp.value == "__main__":
            module_globals["__name__"] = "__main__"
            block = ast.Module(body=node.body, type_ignores=[])
            exec(compile(block, str(script_path), "exec"), module_globals)
            return
    raise RuntimeError(f"No __main__ block in {script_path}")


def run_mertools_script(script: str) -> None:
    script_path = Path(script).resolve()
    if (
        os.environ.get("TMX_INFERENCE_HUMAN") == "1"
        and script_path.name == "inference_hybird.py"
    ):
        module = _load_script_module(script_path)
        from src.inference.human_inference_hook import patch_human_get_name2cls

        patch_human_get_name2cls(module)
        _exec_script_main(script_path, module.__dict__)
        return

    if script_path.name == "train.py" and os.environ.get("TMX_HUMAN_TRAIN_HOLDOUT") == "1":
        from src.inference.human_inference_hook import install_human_train_holdout_hook

        install_human_train_holdout_hook()

    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.mertools_entry <script.py> [args...]")
        sys.exit(1)

    apply_compat_shims()
    apply_torch_amp_compat()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    run_mertools_script(script)


if __name__ == "__main__":
    main()
