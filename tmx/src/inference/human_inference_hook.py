"""Human 数据集推理 hook — 扩展官方 inference_hybird.get_name2cls。"""

from __future__ import annotations


def install_human_inference_hook() -> None:
    import inference_hybird as inf
    from my_affectgpt.datasets.datasets.human_dataset import Human_Dataset

    if getattr(inf.get_name2cls, "_tmx_human_hook", False):
        return

    original = inf.get_name2cls

    def get_name2cls(dataset: str):
        if dataset == "Human":
            return Human_Dataset()
        return original(dataset)

    get_name2cls._tmx_human_hook = True  # type: ignore[attr-defined]
    inf.get_name2cls = get_name2cls
