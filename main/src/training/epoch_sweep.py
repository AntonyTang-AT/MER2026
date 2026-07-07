"""推理 epoch 扫描 — 阶段 5.6。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.core.config_loader import get_project_root, load_yaml
from src.evaluation.baseline_report import evaluate_openset_npz
from src.inference.affectgpt_runner import run_inference
from src.inference.openset_extractor import openset_path_for_reason, run_ovlabel_extraction
from src.training.mertools_paths import get_paths, sync_mertools_config


CHECKPOINT_RE = re.compile(r"checkpoint_(\d{6})_loss_")


@dataclass
class EpochSweepRow:
    epoch: int
    checkpoint: str
    ew_f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    openset_npz: str | None = None
    status: str = "pending"
    notes: str = ""


def parse_epoch_from_checkpoint(path: Path) -> int | None:
    match = CHECKPOINT_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def find_train_cfg_name(train_run: str) -> str:
    bl = load_yaml("baseline.yaml")["train"]
    if train_run not in bl:
        raise ValueError(f"Unknown train run: {train_run}")
    cfg = bl[train_run]["cfg"]
    return Path(cfg).stem


def find_ckpt_roots(mertools_root: Path, train_run: str) -> list[Path]:
    cfg_name = find_train_cfg_name(train_run)
    roots_dir = mertools_root / "output" / cfg_name
    if not roots_dir.is_dir():
        return []
    return sorted(
        [p for p in roots_dir.iterdir() if p.is_dir() and p.name.startswith(cfg_name)],
        key=lambda p: p.stat().st_mtime,
    )


def select_ckpt_root(mertools_root: Path, train_run: str, ckpt_root: Path | None = None) -> Path:
    if ckpt_root is not None:
        path = ckpt_root if ckpt_root.is_absolute() else mertools_root / ckpt_root
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint root not found: {path}")
        return path

    roots = find_ckpt_roots(mertools_root, train_run)
    if not roots:
        raise FileNotFoundError(f"No checkpoint roots found for train run: {train_run}")

    best = max(
        roots,
        key=lambda root: (
            len(list(root.glob("checkpoint_*.pth"))),
            root.stat().st_mtime,
        ),
    )
    return best


def list_checkpoints(ckpt_root: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in sorted(ckpt_root.glob("checkpoint_*.pth")):
        epoch = parse_epoch_from_checkpoint(path)
        if epoch is not None:
            mapping[epoch] = path
    return mapping


def plan_epochs(
    *,
    start: int,
    end: int,
    skip: int,
    available: set[int] | None = None,
) -> list[int]:
    epochs = [epoch for epoch in range(start, end + 1) if epoch % skip == 0]
    if available is not None:
        epochs = [epoch for epoch in epochs if epoch in available]
    return epochs


def assign_epochs_to_gpus(epochs: list[int], gpus: list[str]) -> dict[str, list[int]]:
    """Round-robin 将 epoch 分配到各 GPU。"""
    if not gpus:
        raise ValueError("At least one GPU id required")
    buckets: dict[str, list[int]] = {gpu: [] for gpu in gpus}
    for index, epoch in enumerate(epochs):
        buckets[gpus[index % len(gpus)]].append(epoch)
    return buckets


def parse_gpus(value: str | None) -> list[str] | None:
    if value is None:
        return None
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("Empty --gpus value")
    return gpus


def _resolve_cfg_path(train_run: str) -> str:
    bl = load_yaml("baseline.yaml")["train"]
    return bl[train_run]["cfg"]


def _infer_dataset_for_split(split: str) -> str:
    if split == "val":
        return "Human"
    return load_yaml("baseline.yaml")["inference"]["dataset"]


def _epoch_artifact_paths(
    epoch: int,
    *,
    ckpt_root: Path,
    train_run: str,
    split: str,
) -> tuple[Path, Path, Path, str] | None:
    ckpt_path = list_checkpoints(ckpt_root).get(epoch)
    if ckpt_path is None:
        return None
    dataset = _infer_dataset_for_split(split)
    bl = load_yaml("baseline.yaml")
    base_root = bl["inference"]["base_root"]
    save_root = (
        get_paths()["mertools_root"]
        / f"{base_root}-{dataset.lower()}"
        / ckpt_root.name
    )
    epoch_tag = ckpt_path.stem
    reason_npz = save_root / f"{epoch_tag}.npz"
    openset_npz = save_root / f"{epoch_tag}-openset.npz"
    return ckpt_path, reason_npz, openset_npz, dataset


def run_epoch_inference_only(
    epoch: int,
    *,
    train_run: str,
    ckpt_root: Path,
    cuda_devices: str | None = None,
    skip_existing: bool = True,
) -> EpochSweepRow:
    cfg_path = _resolve_cfg_path(train_run)
    artifacts = _epoch_artifact_paths(
        epoch, ckpt_root=ckpt_root, train_run=train_run, split="val"
    )
    if artifacts is None:
        return EpochSweepRow(
            epoch=epoch,
            checkpoint="",
            status="missing_checkpoint",
            notes=f"No checkpoint for epoch {epoch}",
        )

    ckpt_path, reason_npz, openset_npz, dataset = artifacts
    row = EpochSweepRow(epoch=epoch, checkpoint=str(ckpt_path), status="running")

    if skip_existing and openset_npz.is_file():
        row.openset_npz = str(openset_npz)
        row.status = "skipped_existing"
        return row

    if skip_existing and reason_npz.is_file():
        row.status = "infer_done"
        row.notes = "reason npz exists; skip inference"
        return row

    bl = load_yaml("baseline.yaml")
    base_root = bl["inference"]["base_root"]
    options = [
        f"inference.ckpt_root={ckpt_root.as_posix()}",
        f"inference.test_epoch={epoch}",
        f"inference.base_root={base_root}",
        f"inference.face_or_frame={bl['inference']['face_or_frame']}",
        f"inference.gpu={bl['inference']['gpu']}",
    ]
    code = run_inference(
        cfg_path=cfg_path,
        dataset=dataset,
        zeroshot=True,
        options=options,
        cuda_devices=cuda_devices,
    )
    if code != 0:
        row.status = "infer_failed"
        row.notes = f"inference exit code {code}"
        return row

    if not reason_npz.is_file():
        row.status = "reason_missing"
        row.notes = f"Expected reason npz: {reason_npz}"
        return row

    row.status = "infer_done"
    return row


def run_epoch_ovlabel_eval(
    row: EpochSweepRow,
    *,
    train_run: str,
    ckpt_root: Path,
    split: str = "val",
    cuda_devices: str | None = None,
    skip_existing: bool = True,
) -> EpochSweepRow:
    if row.status in {"missing_checkpoint", "infer_failed", "reason_missing"}:
        return row

    artifacts = _epoch_artifact_paths(
        row.epoch, ckpt_root=ckpt_root, train_run=train_run, split=split
    )
    if artifacts is None:
        return row

    _ckpt_path, reason_npz, openset_npz, _dataset = artifacts

    if row.status != "skipped_existing":
        if skip_existing and openset_npz.is_file():
            row.openset_npz = str(openset_npz)
            row.status = "skipped_existing"
        elif not reason_npz.is_file():
            row.status = "reason_missing"
            row.notes = f"Expected reason npz: {reason_npz}"
            return row
        else:
            ov_code = run_ovlabel_extraction(
                reason_npz=reason_npz,
                store_npz=openset_path_for_reason(reason_npz),
                cuda_devices=cuda_devices,
            )
            if ov_code != 0:
                row.status = "ovlabel_failed"
                row.notes = f"ovlabel exit code {ov_code}"
                return row
            if not openset_npz.is_file():
                row.status = "openset_missing"
                return row
            row.openset_npz = str(openset_npz)
            row.status = "inferred"

    if split == "val" and row.openset_npz:
        record = evaluate_openset_npz(
            Path(row.openset_npz),
            split="val",
            model_name=f"{train_run}_epoch{row.epoch}",
        )
        row.ew_f1 = record.ew_f1
        row.precision = record.precision
        row.recall = record.recall
        row.status = "evaluated"

    return row


def run_single_epoch(
    epoch: int,
    *,
    train_run: str,
    ckpt_root: Path,
    split: str = "val",
    cuda_devices: str | None = None,
    skip_existing: bool = True,
) -> EpochSweepRow:
    row = run_epoch_inference_only(
        epoch,
        train_run=train_run,
        ckpt_root=ckpt_root,
        cuda_devices=cuda_devices,
        skip_existing=skip_existing,
    )
    return run_epoch_ovlabel_eval(
        row,
        train_run=train_run,
        ckpt_root=ckpt_root,
        split=split,
        cuda_devices=cuda_devices,
        skip_existing=skip_existing,
    )


def _write_progress_row(
    row: EpochSweepRow,
    *,
    progress_dir: Path | None,
    tag: str,
) -> None:
    if progress_dir is None:
        return
    progress_path = progress_dir / f"epoch_{row.epoch:03d}_{tag}.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(asdict(row), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[{tag}] epoch {row.epoch}: {row.status} ew_f1={row.ew_f1}", flush=True)


def _run_inference_on_gpu(
    gpu: str,
    epochs: list[int],
    *,
    train_run: str,
    ckpt_root: str,
    skip_existing: bool,
    progress_dir: str | None,
) -> list[dict]:
    """单 GPU worker：仅 AffectGPT 推理，不加载 vLLM。"""
    root = Path(ckpt_root)
    progress_path = Path(progress_dir) if progress_dir else None
    rows: list[EpochSweepRow] = []
    for epoch in epochs:
        row = run_epoch_inference_only(
            epoch,
            train_run=train_run,
            ckpt_root=root,
            cuda_devices=gpu,
            skip_existing=skip_existing,
        )
        rows.append(row)
        _write_progress_row(row, progress_dir=progress_path, tag=f"infer_gpu{gpu}")
    return [asdict(row) for row in rows]


def _run_ovlabel_epochs_on_gpu(
    gpu: str,
    row_dicts: list[dict],
    *,
    train_run: str,
    ckpt_root: str,
    split: str,
    skip_existing: bool,
    progress_dir: str | None,
) -> list[dict]:
    """单 GPU worker：串行 ovlabel + eval（各 worker 独立 vLLM 实例）。"""
    root = Path(ckpt_root)
    progress_path = Path(progress_dir) if progress_dir else None
    finalized: list[EpochSweepRow] = []
    for item in row_dicts:
        row = EpochSweepRow(**item)
        row = run_epoch_ovlabel_eval(
            row,
            train_run=train_run,
            ckpt_root=root,
            split=split,
            cuda_devices=gpu,
            skip_existing=skip_existing,
        )
        finalized.append(row)
        _write_progress_row(row, progress_dir=progress_path, tag=f"eval_gpu{gpu}")
    return [asdict(row) for row in finalized]


def run_epoch_sweep(
    *,
    train_run: str = "human",
    start: int = 10,
    end: int = 60,
    skip: int = 5,
    split: str = "val",
    ckpt_root: Path | None = None,
    cuda_devices: str | None = None,
    gpus: list[str] | None = None,
    ovlabel_gpu: str | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
    progress_dir: Path | None = None,
) -> list[EpochSweepRow]:
    sync_mertools_config(verbose=False)
    paths = get_paths()
    root = select_ckpt_root(paths["mertools_root"], train_run, ckpt_root)
    available = set(list_checkpoints(root).keys())
    epochs = plan_epochs(start=start, end=end, skip=skip, available=available)

    if dry_run:
        rows: list[EpochSweepRow] = []
        for epoch in epochs:
            ckpt = list_checkpoints(root).get(epoch)
            rows.append(
                EpochSweepRow(
                    epoch=epoch,
                    checkpoint=str(ckpt) if ckpt else "",
                    status="planned" if ckpt else "missing_checkpoint",
                )
            )
        return rows

    active_gpus = gpus
    if active_gpus is None and cuda_devices is not None:
        active_gpus = parse_gpus(cuda_devices)
    if active_gpus is None:
        active_gpus = ["0"]

    if len(active_gpus) == 1 or len(epochs) <= 1:
        gpu = active_gpus[0]
        return [
            run_single_epoch(
                epoch,
                train_run=train_run,
                ckpt_root=root,
                split=split,
                cuda_devices=gpu,
                skip_existing=skip_existing,
            )
            for epoch in epochs
        ]

    assignment = assign_epochs_to_gpus(epochs, active_gpus)
    progress_path = progress_dir or get_project_root() / "logs" / "epoch_sweep" / train_run
    progress_path.mkdir(parents=True, exist_ok=True)

    print(
        "Multi-GPU epoch sweep (infer parallel + ovlabel parallel):",
        ", ".join(f"gpu{gpu}={eps}" for gpu, eps in assignment.items() if eps),
        flush=True,
    )

    row_dicts: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(active_gpus)) as pool:
        futures = {
            pool.submit(
                _run_inference_on_gpu,
                gpu,
                gpu_epochs,
                train_run=train_run,
                ckpt_root=str(root),
                skip_existing=skip_existing,
                progress_dir=str(progress_path),
            ): gpu
            for gpu, gpu_epochs in assignment.items()
            if gpu_epochs
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                row_dicts.extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"GPU {gpu} inference worker failed: {exc}") from exc

    rows = [EpochSweepRow(**item) for item in row_dicts]
    rows.sort(key=lambda row: row.epoch)

    ovlabel_gpus = parse_gpus(ovlabel_gpu) if ovlabel_gpu else active_gpus
    ovlabel_assignment = assign_epochs_to_gpus([row.epoch for row in rows], ovlabel_gpus)

    print(
        "Phase 2: ovlabel + eval parallel:",
        ", ".join(f"gpu{gpu}={eps}" for gpu, eps in ovlabel_assignment.items() if eps),
        flush=True,
    )

    finalized_dicts: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(ovlabel_gpus)) as pool:
        futures = {
            pool.submit(
                _run_ovlabel_epochs_on_gpu,
                gpu,
                [asdict(row) for row in rows if row.epoch in gpu_epochs],
                train_run=train_run,
                ckpt_root=str(root),
                split=split,
                skip_existing=skip_existing,
                progress_dir=str(progress_path),
            ): gpu
            for gpu, gpu_epochs in ovlabel_assignment.items()
            if gpu_epochs
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                finalized_dicts.extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"GPU {gpu} ovlabel worker failed: {exc}") from exc

    finalized = [EpochSweepRow(**item) for item in finalized_dicts]
    finalized.sort(key=lambda row: row.epoch)
    return finalized


def write_sweep_report(
    rows: list[EpochSweepRow],
    *,
    train_run: str,
    out_md: Path | None = None,
    out_json: Path | None = None,
) -> tuple[Path, Path]:
    project = get_project_root()
    out_md = out_md or project / "docs/reports" / f"epoch_sweep_{train_run}.md"
    out_json = out_json or project / "docs/reports" / f"epoch_sweep_{train_run}.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    evaluated = [row for row in rows if row.ew_f1 is not None]
    best = max(evaluated, key=lambda row: row.ew_f1 or -1.0) if evaluated else None

    md_lines = [
        f"# Epoch Sweep — {train_run}",
        "",
        f"- 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本数: {len(rows)}",
        "",
        "| Epoch | EW-F1 | Precision | Recall | Status | Checkpoint |",
        "|------:|------:|----------:|-------:|--------|------------|",
    ]
    for row in rows:
        ew = f"{row.ew_f1:.4f}" if row.ew_f1 is not None else "—"
        prec = f"{row.precision:.4f}" if row.precision is not None else "—"
        rec = f"{row.recall:.4f}" if row.recall is not None else "—"
        ckpt = Path(row.checkpoint).name if row.checkpoint else "—"
        md_lines.append(
            f"| {row.epoch} | {ew} | {prec} | {rec} | {row.status} | {ckpt} |"
        )

    if best is not None:
        md_lines.extend(
            [
                "",
                "## Best",
                "",
                f"- epoch: **{best.epoch}**",
                f"- EW-F1: **{best.ew_f1:.4f}**",
                f"- openset: `{best.openset_npz}`",
            ]
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    payload = {
        "train_run": train_run,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": [asdict(row) for row in rows],
        "best_epoch": asdict(best) if best else None,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if best is not None:
        best_path = project / "experiments" / "exp001_baseline" / "best_epoch.json"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(
            json.dumps(
                {
                    "epoch": best.epoch,
                    "ew_f1": best.ew_f1,
                    "checkpoint": best.checkpoint,
                    "openset_npz": best.openset_npz,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return out_md, out_json


def write_summary_csv(rows: list[EpochSweepRow], out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "ew_f1",
                "precision",
                "recall",
                "status",
                "checkpoint",
                "openset_npz",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": row.epoch,
                    "ew_f1": row.ew_f1,
                    "precision": row.precision,
                    "recall": row.recall,
                    "status": row.status,
                    "checkpoint": row.checkpoint,
                    "openset_npz": row.openset_npz,
                    "notes": row.notes,
                }
            )
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep inference epochs and evaluate val EW-F1")
    parser.add_argument("--train-run", default="human", help="baseline.yaml train key")
    parser.add_argument("--epochs", default="10-60", help="Epoch range start-end")
    parser.add_argument("--skip", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--split", default="val", choices=["val", "train", "all"])
    parser.add_argument("--ckpt-root", default=None, help="Override checkpoint directory")
    parser.add_argument("--cuda", default=None, help="Single GPU id (legacy alias for --gpus)")
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for parallel inference (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--ovlabel-gpu",
        default=None,
        help="Comma-separated GPU ids for parallel ovlabel (default: same as --gpus)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    start_s, end_s = args.epochs.split("-", 1)
    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else None
    gpus = parse_gpus(args.gpus or args.cuda or "0,1,2,3,4")

    rows = run_epoch_sweep(
        train_run=args.train_run,
        start=int(start_s),
        end=int(end_s),
        skip=args.skip,
        split=args.split,
        ckpt_root=ckpt_root,
        gpus=gpus,
        ovlabel_gpu=args.ovlabel_gpu,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip_existing,
    )

    out_md, out_json = write_sweep_report(rows, train_run=args.train_run)
    csv_path = write_summary_csv(
        rows,
        get_project_root() / "experiments" / "exp001_baseline" / "epoch_sweep.csv",
    )
    print(f"Report: {out_md}")
    print(f"JSON: {out_json}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
