"""调用官方 train.py — AffectGPT 训练封装。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.core.config_loader import get_project_root, load_yaml
from src.training.mertools_paths import (
    get_paths,
    patch_mercaptionplus_label_csv,
    restore_mercaptionplus_label_csv,
    sync_mertools_config,
    verify_mertools_config,
)

_TRAIN_TARGETS = ("human", "mercaptionplus", "mixed")


def _baseline_cfg(name: str = "human") -> str:
    bl = load_yaml("baseline.yaml")
    if name not in _TRAIN_TARGETS:
        raise ValueError(f"Unknown train target: {name}")
    return bl["train"][name]["cfg"]


def _maybe_use_filtered_mercaptionplus(*, use_filtered: bool) -> None:
    if not use_filtered:
        restore_mercaptionplus_label_csv(verbose=True)
        return
    bl = load_yaml("baseline.yaml")
    label_name = bl["train"]["mercaptionplus"].get(
        "filtered_label_csv",
        "track2_train_mercaptionplus_filtered.csv",
    )
    paths = get_paths()
    label_path = paths["data_root"] / label_name
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Filtered MER-Caption+ CSV not found: {label_path}. "
            "Run: bash scripts/filter_mercaptionplus.sh"
        )
    patch_mercaptionplus_label_csv(label_path, verbose=True)


def run_train(
    cfg_path: str | None = None,
    *,
    target: str = "human",
    extra_args: list[str] | None = None,
    cuda_devices: str | None = None,
    use_filtered_mercaptionplus: bool = False,
) -> int:
    sync_mertools_config(verbose=True)
    if target in ("mercaptionplus", "mixed") and use_filtered_mercaptionplus:
        _maybe_use_filtered_mercaptionplus(use_filtered=True)
    elif target == "human":
        restore_mercaptionplus_label_csv(verbose=False)

    issues = verify_mertools_config()
    if issues:
        print("Config verification failed:", file=sys.stderr)
        for item in issues:
            print(f"  - {item}", file=sys.stderr)
        return 1

    paths = get_paths()
    mertools_root = paths["mertools_root"]
    cfg = cfg_path or _baseline_cfg(target)
    project_root = paths["project_root"]

    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    if target in ("human", "mixed"):
        env["TMX_HUMAN_TRAIN_HOLDOUT"] = "1"
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.training.mertools_entry",
        "train.py",
        f"--cfg-path={cfg}",
        *(extra_args or []),
    ]
    print("Running:", " ".join(cmd), f"(cwd={mertools_root})")
    return subprocess.call(cmd, cwd=str(mertools_root), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AffectGPT via official train.py")
    parser.add_argument(
        "--target",
        choices=list(_TRAIN_TARGETS),
        default="human",
        help="Which baseline config to use",
    )
    parser.add_argument("--cfg-path", default=None, help="Override train config yaml")
    parser.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--use-filtered",
        action="store_true",
        help="Use track2_train_mercaptionplus_filtered.csv for MER-Caption+ label path",
    )
    parser.add_argument("--sync-only", action="store_true", help="Only sync MERTools paths")
    parser.add_argument("extra", nargs="*", help="Extra args passed to train.py")
    args = parser.parse_args()

    if args.sync_only:
        sync_mertools_config(verbose=True)
        issues = verify_mertools_config()
        if issues:
            for item in issues:
                print(f"ISSUE: {item}")
            sys.exit(1)
        print("OK: MERTools config synced and verified")
        return

    code = run_train(
        cfg_path=args.cfg_path,
        target=args.target,
        extra_args=args.extra,
        cuda_devices=args.cuda,
        use_filtered_mercaptionplus=args.use_filtered,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
