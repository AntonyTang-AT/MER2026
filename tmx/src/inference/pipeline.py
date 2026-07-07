"""主流水线 — Stage A→B→C→提交 编排。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from src.core.config_loader import get_project_root, load_yaml
from src.data.dataset_index import MER2026Index
from src.data.submission_formatter import format_submission
from src.evaluation.eval_runner import run_eval
from src.evaluation.mertools_bridge import load_npz_predictions
from src.inference.affectgpt_runner import find_latest_openset_npz, run_stage_b
from src.inference.ensemble import ensemble_and_save
from src.inference.openset_extractor import extract_openset_custom, openset_path_for_reason
from src.inference.stage_manager import StageManager, StagePlan
from src.routing.run_routing import run_batch
from src.training.mertools_paths import sync_mertools_config


@dataclass
class PipelineArtifacts:
    routing_json: Path | None = None
    reason_npz: Path | None = None
    openset_npz: Path | None = None
    submission_csv: Path | None = None
    eval_json: Path | None = None
    timing: dict[str, float] = field(default_factory=dict)
    mode: str = "submit"


@contextmanager
def timed_stage(timing: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timing[name] = round(time.perf_counter() - start, 3)


class PipelineRunner:
    """Stage A→B→C→(ensemble)→submission/eval 主流水线。"""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config if config is not None else load_yaml("pipeline.yaml")
        self.project_root = get_project_root()
        self.stage_manager = StageManager(self.config)

    def _routing_json_path(self, split: str) -> Path:
        artifacts = self.config.get("artifacts") or {}
        routing_dir = artifacts.get("routing_dir", "outputs/routing")
        return (self.project_root / routing_dir / f"{split}_routing.json").resolve()

    def _submission_dir(self) -> Path:
        artifacts = self.config.get("artifacts") or {}
        sub_dir = artifacts.get("submission_dir", "outputs/submissions")
        return (self.project_root / sub_dir).resolve()

    def _timing_dir(self) -> Path:
        artifacts = self.config.get("artifacts") or {}
        timing_dir = artifacts.get("timing_dir", "outputs/pipeline_runs")
        return (self.project_root / timing_dir).resolve()

    def run_stage_routing(
        self,
        plan: StagePlan,
        *,
        limit: int | None,
        skip_existing: bool,
        dry_run: bool,
        routing_json: Path | None,
        timing: dict[str, float],
    ) -> Path | None:
        if not plan.routing:
            return routing_json

        json_path = routing_json or self._routing_json_path(plan.routing_split)
        if skip_existing and json_path.is_file():
            print(f"Skip routing (exists): {json_path}")
            return json_path

        if dry_run:
            print(f"[dry-run] routing -> {json_path}")
            return json_path

        with timed_stage(timing, "routing"):
            index = MER2026Index.from_config()
            samples = index.load_split(plan.routing_split, limit=limit)
            if not samples:
                raise RuntimeError(f"No samples for routing split={plan.routing_split}")
            csv_path = json_path.with_suffix(".csv")
            run_batch(samples, output_json=json_path, output_csv=csv_path)
            print(f"Routing done: {json_path} ({len(samples)} samples)")
        return json_path

    def run_stage_affectgpt(
        self,
        plan: StagePlan,
        *,
        routing_json: Path | None,
        limit: int | None,
        cuda_devices: str | None,
        dry_run: bool,
        timing: dict[str, float],
    ) -> Path | None:
        if not plan.affectgpt:
            return None

        if dry_run:
            print("[dry-run] affectgpt inference")
            return None

        affectgpt_cfg = (self.config.get("stages") or {}).get("affectgpt") or {}
        dataset = load_yaml("baseline.yaml")["inference"]["dataset"]

        with timed_stage(timing, "affectgpt"):
            reason_npz = run_stage_b(
                use_routing=plan.use_routing_in_prompt,
                routing_json=routing_json,
                prompt_variant=plan.prompt_variant,
                dataset=dataset,
                zeroshot=bool(affectgpt_cfg.get("zeroshot", True)),
                cuda_devices=cuda_devices,
                limit=limit,
                dry_run=False,
            )
        print(f"Stage B reason npz: {reason_npz}")
        return reason_npz

    def run_stage_openset(
        self,
        plan: StagePlan,
        *,
        reason_npz: Path | None,
        cuda_devices: str | None,
        skip_existing: bool,
        dry_run: bool,
        timing: dict[str, float],
    ) -> Path | None:
        if not plan.openset:
            return None

        if reason_npz is None:
            raise ValueError("openset_extract enabled but reason_npz is missing")

        store_npz = openset_path_for_reason(reason_npz)
        if skip_existing and store_npz.is_file():
            print(f"Skip openset (exists): {store_npz}")
            return store_npz

        if dry_run:
            print(f"[dry-run] openset -> {store_npz}")
            return store_npz

        openset_cfg = (self.config.get("stages") or {}).get("openset_extract") or {}
        batch_size = int(openset_cfg.get("batch_size", 8))

        with timed_stage(timing, "openset_extract"):
            code = extract_openset_custom(
                reason_npz=reason_npz,
                store_npz=store_npz,
                cuda_devices=cuda_devices,
                prompt_variant=plan.openset_prompt_variant,
                postprocess_mode=plan.openset_postprocess,
                batch_size=batch_size,
            )
            if code != 0:
                raise RuntimeError(f"Openset extraction failed with exit code {code}")
        print(f"Stage C openset npz: {store_npz}")
        return store_npz

    def run_stage_ensemble(
        self,
        plan: StagePlan,
        *,
        openset_npz: Path | None,
        timing: dict[str, float],
    ) -> Path | None:
        if not plan.ensemble or openset_npz is None:
            return openset_npz

        # 初版：若仅有一个 npz，与自身 union 等价；多 npz 由 glob 同 stem 前缀收集
        parent = openset_npz.parent
        stem_prefix = openset_npz.name.replace("-openset.npz", "")
        siblings = sorted(parent.glob(f"{stem_prefix}*-openset.npz"))
        if len(siblings) <= 1:
            all_npz = list(parent.glob("*-openset.npz"))
            siblings = sorted(all_npz)

        out_npz = parent / f"{stem_prefix}-ensemble-openset.npz"
        with timed_stage(timing, "ensemble"):
            ensemble_and_save(siblings, out_npz, strategy=plan.ensemble_strategy)
        print(f"Ensemble openset: {out_npz} (from {len(siblings)} files)")
        return out_npz

    def run_stage_submission(
        self,
        plan: StagePlan,
        *,
        openset_npz: Path | None,
        submission_out: Path | None,
        dry_run: bool,
        timing: dict[str, float],
    ) -> Path | None:
        if not plan.submission:
            return None
        if openset_npz is None:
            raise ValueError("submission enabled but openset_npz is missing")

        if dry_run:
            out = submission_out or self._submission_dir() / "track2_dryrun.csv"
            with timed_stage(timing, "submission"):
                print(f"[dry-run] submission -> {out}")
            return out

        with timed_stage(timing, "submission"):
            name2pred = load_npz_predictions(openset_npz)
            out = format_submission(name2pred, out_path=submission_out)
        print(f"Submission CSV: {out}")
        return out

    def run_stage_evaluation(
        self,
        plan: StagePlan,
        *,
        openset_npz: Path | None,
        eval_out: Path | None,
        dry_run: bool,
        timing: dict[str, float],
    ) -> Path | None:
        if not plan.evaluation or openset_npz is None:
            return None

        if dry_run:
            out = eval_out or self._timing_dir() / "eval_dryrun.json"
            print(f"[dry-run] evaluation -> {out}")
            return out

        with timed_stage(timing, "evaluation"):
            payload = run_eval(
                openset_npz,
                split=plan.eval_split,
                out=eval_out,
            )
        out_path = eval_out or self._timing_dir() / f"eval_{plan.eval_split}.json"
        if eval_out is None and payload:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Evaluation saved: {out_path}")
        return out_path

    def run(
        self,
        *,
        mode: str = "submit",
        limit: int | None = None,
        skip_existing: bool | None = None,
        dry_run: bool | None = None,
        cuda_devices: str | None = None,
        routing_json: Path | str | None = None,
        reason_npz: Path | str | None = None,
        openset_npz: Path | str | None = None,
        submission_out: Path | str | None = None,
        skip_stages: list[str] | None = None,
        profile: bool = False,
    ) -> PipelineArtifacts:
        runtime = self.config.get("runtime") or {}
        skip_existing = runtime.get("skip_existing", True) if skip_existing is None else skip_existing
        dry_run = runtime.get("dry_run", False) if dry_run is None else dry_run

        plan = self.stage_manager.build_plan(mode=mode)
        if skip_stages:
            for name in skip_stages:
                if name == "routing":
                    plan.routing = False
                elif name == "affectgpt":
                    plan.affectgpt = False
                elif name in ("openset", "openset_extract"):
                    plan.openset = False
                elif name == "ensemble":
                    plan.ensemble = False
                elif name == "submission":
                    plan.submission = False
                elif name == "evaluation":
                    plan.evaluation = False

        sync_mertools_config(verbose=False)

        artifacts = PipelineArtifacts(mode=mode)
        timing: dict[str, float] = {}
        routing_path = Path(routing_json) if routing_json else None
        reason_path = Path(reason_npz) if reason_npz else None
        openset_path = Path(openset_npz) if openset_npz else None

        print("Pipeline plan:", self.stage_manager.enabled_stage_names(plan))

        artifacts.routing_json = self.run_stage_routing(
            plan,
            limit=limit,
            skip_existing=skip_existing,
            dry_run=dry_run,
            routing_json=routing_path,
            timing=timing,
        )

        if plan.affectgpt and reason_path is None:
            reason_path = self.run_stage_affectgpt(
                plan,
                routing_json=artifacts.routing_json,
                limit=limit,
                cuda_devices=cuda_devices,
                dry_run=dry_run,
                timing=timing,
            )
        artifacts.reason_npz = reason_path

        if plan.openset and openset_path is None and reason_path is not None:
            openset_path = self.run_stage_openset(
                plan,
                reason_npz=reason_path,
                cuda_devices=cuda_devices,
                skip_existing=skip_existing,
                dry_run=dry_run,
                timing=timing,
            )
        elif openset_path is None and reason_path is not None:
            openset_path = find_latest_openset_npz(reason_path)

        if plan.ensemble and openset_path is not None:
            openset_path = self.run_stage_ensemble(plan, openset_npz=openset_path, timing=timing)

        artifacts.openset_npz = openset_path

        sub_out = Path(submission_out) if submission_out else None
        artifacts.submission_csv = self.run_stage_submission(
            plan,
            openset_npz=openset_path,
            submission_out=sub_out,
            dry_run=dry_run,
            timing=timing,
        )

        eval_out = self._timing_dir() / f"eval_{plan.eval_split}_{datetime.now():%Y%m%d_%H%M%S}.json"
        artifacts.eval_json = self.run_stage_evaluation(
            plan,
            openset_npz=openset_path,
            eval_out=eval_out if plan.evaluation else None,
            dry_run=dry_run,
            timing=timing,
        )

        if timing:
            timing["total"] = round(sum(v for k, v in timing.items() if k != "total"), 3)
        artifacts.timing = timing

        if profile or timing:
            self._timing_dir().mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            timing_path = self._timing_dir() / f"timing_{mode}_{ts}.json"
            timing_path.write_text(
                json.dumps(
                    {
                        "mode": mode,
                        "limit": limit,
                        "dry_run": dry_run,
                        "stages": self.stage_manager.enabled_stage_names(plan),
                        "artifacts": {
                            "routing_json": str(artifacts.routing_json) if artifacts.routing_json else None,
                            "reason_npz": str(artifacts.reason_npz) if artifacts.reason_npz else None,
                            "openset_npz": str(artifacts.openset_npz) if artifacts.openset_npz else None,
                            "submission_csv": str(artifacts.submission_csv) if artifacts.submission_csv else None,
                        },
                        "timing": timing,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Timing saved: {timing_path}")

        return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MER2026 full inference pipeline")
    parser.add_argument("--mode", choices=["submit", "eval"], default="submit")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--routing-json", type=Path, default=None)
    parser.add_argument("--reason-npz", type=Path, default=None)
    parser.add_argument("--openset-npz", type=Path, default=None)
    parser.add_argument("--submission-out", type=Path, default=None)
    parser.add_argument(
        "--skip-stages",
        default="",
        help="Comma-separated: routing,affectgpt,openset,ensemble,submission,evaluation",
    )
    args = parser.parse_args(argv)

    skip_existing = True
    if args.no_skip_existing:
        skip_existing = False
    elif args.skip_existing:
        skip_existing = True

    skip_stages = [s.strip() for s in args.skip_stages.split(",") if s.strip()]

    runner = PipelineRunner()
    try:
        runner.run(
            mode=args.mode,
            limit=args.limit,
            skip_existing=skip_existing,
            dry_run=args.dry_run,
            cuda_devices=args.cuda,
            routing_json=args.routing_json,
            reason_npz=args.reason_npz,
            openset_npz=args.openset_npz,
            submission_out=args.submission_out,
            skip_stages=skip_stages or None,
            profile=args.profile,
        )
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
