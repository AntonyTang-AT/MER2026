"""Human 数据集 hook — 推理 val 列表与训练 hold-out。"""

from __future__ import annotations

import os
from types import ModuleType

from src.data.human_ov_split import filter_train_annotations


def _patch_human_dataset_cls() -> None:
    from my_affectgpt.datasets.datasets.human_dataset import Human_Dataset

    if getattr(Human_Dataset, "_tmx_read_test_names", False):
        return

    def read_test_names(self):
        val_list = os.environ.get("TMX_HUMAN_VAL_LIST")
        if val_list and os.path.isfile(val_list):
            with open(val_list, encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
            if names:
                return names
        return [item["name"] for item in self.annotation]

    Human_Dataset.read_test_names = read_test_names  # type: ignore[method-assign]
    Human_Dataset._tmx_read_test_names = True


def patch_human_get_name2cls(module: ModuleType) -> None:
    """在 inference_hybird 模块加载后、执行 main 前注入 Human 数据集。"""
    if getattr(module.get_name2cls, "_tmx_human_hook", False):
        return

    _patch_human_dataset_cls()
    from my_affectgpt.datasets.datasets.human_dataset import Human_Dataset

    original = module.get_name2cls

    def get_name2cls(dataset: str):
        if dataset == "Human":
            return Human_Dataset()
        return original(dataset)

    get_name2cls._tmx_human_hook = True  # type: ignore[attr-defined]
    module.get_name2cls = get_name2cls


def install_human_inference_hook() -> None:
    """兼容旧入口：若 inference_hybird 已导入则直接 patch。"""
    import sys

    module = sys.modules.get("inference_hybird")
    if module is not None:
        patch_human_get_name2cls(module)


def install_human_train_holdout_hook() -> None:
    """训练时 Human_Dataset 排除 val（TMX_HUMAN_TRAIN_HOLDOUT=1）。"""
    if os.environ.get("TMX_HUMAN_TRAIN_HOLDOUT") != "1":
        return

    from my_affectgpt.datasets.datasets.human_dataset import Human_Dataset

    if getattr(Human_Dataset, "_tmx_train_holdout", False):
        return

    original_init = Human_Dataset.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        before = len(self.annotation)
        self.annotation = filter_train_annotations(self.annotation)
        after = len(self.annotation)
        print(
            f"[TMX] Human_Dataset train hold-out: {before} -> {after} "
            f"(excluded {before - after} val samples)",
            flush=True,
        )

    Human_Dataset.__init__ = __init__  # type: ignore[method-assign, assignment]
    Human_Dataset._tmx_train_holdout = True
