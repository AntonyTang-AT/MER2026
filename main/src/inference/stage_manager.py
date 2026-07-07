"""流水线阶段管理与依赖校验。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.config_loader import load_yaml


@dataclass
class StagePlan:
    routing: bool
    affectgpt: bool
    openset: bool
    ensemble: bool
    submission: bool
    evaluation: bool
    use_routing_in_prompt: bool
    routing_split: str
    prompt_variant: str
    openset_prompt_variant: str
    openset_postprocess: str
    ensemble_strategy: str
    eval_split: str


class StageManager:
    """解析 pipeline.yaml 并生成阶段执行计划。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_yaml("pipeline.yaml")

    def build_plan(self, *, mode: str = "submit") -> StagePlan:
        stages = self.config.get("stages") or {}
        routing_split = "candidate" if mode == "submit" else "human"

        affectgpt_cfg = stages.get("affectgpt") or {}
        openset_cfg = stages.get("openset_extract") or {}
        ensemble_cfg = stages.get("ensemble") or {}
        eval_cfg = stages.get("evaluation") or {}

        use_routing = bool(affectgpt_cfg.get("use_routing_in_prompt", False))
        routing_enabled = bool(stages.get("routing", {}).get("enabled", True))

        if use_routing and not routing_enabled:
            raise ValueError(
                "affectgpt.use_routing_in_prompt=true but stages.routing.enabled=false"
            )

        eval_enabled = bool(eval_cfg.get("enabled", False))
        if mode == "eval":
            eval_enabled = True

        return StagePlan(
            routing=bool(stages.get("routing", {}).get("enabled", True)),
            affectgpt=bool(affectgpt_cfg.get("enabled", True)),
            openset=bool(openset_cfg.get("enabled", True)),
            ensemble=bool(ensemble_cfg.get("enabled", False)),
            submission=bool(stages.get("submission", {}).get("enabled", True))
            and mode == "submit",
            evaluation=eval_enabled and mode == "eval",
            use_routing_in_prompt=use_routing and routing_enabled,
            routing_split=routing_split,
            prompt_variant=str(affectgpt_cfg.get("prompt_variant", "routing")),
            openset_prompt_variant=str(openset_cfg.get("prompt_variant", "ew_aware")),
            openset_postprocess=str(openset_cfg.get("postprocess", "ew")),
            ensemble_strategy=str(ensemble_cfg.get("strategy", "label_union")),
            eval_split=str(eval_cfg.get("split", "val")),
        )

    def enabled_stage_names(self, plan: StagePlan) -> list[str]:
        names: list[str] = []
        if plan.routing:
            names.append("routing")
        if plan.affectgpt:
            names.append("affectgpt")
        if plan.openset:
            names.append("openset_extract")
        if plan.ensemble:
            names.append("ensemble")
        if plan.submission:
            names.append("submission")
        if plan.evaluation:
            names.append("evaluation")
        return names
