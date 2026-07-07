"""封装官方 inference_hybird.py — AffectGPT 推理。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.core.config_loader import load_yaml
from src.training.mertools_paths import get_paths, sync_mertools_config, verify_mertools_config


def _inference_options_from_baseline() -> list[str]:
    bl = load_yaml("baseline.yaml")["inference"]
    opts = [
        f"inference.face_or_frame={bl['face_or_frame']}",
        f"inference.base_root={bl['base_root']}",
        f"inference.test_epochs={bl['test_epochs']}",
        f"inference.skip_epoch={bl['skip_epoch']}",
        f"inference.gpu={bl['gpu']}",
    ]
    if bl.get("ckpt_root"):
        opts.append(f"inference.ckpt_root={bl['ckpt_root']}")
    if bl.get("ckpt_name"):
        opts.append(f"inference.ckpt_name={bl['ckpt_name']}")
    return opts


def run_inference(
    *,
    cfg_path: str | None = None,
    dataset: str | None = None,
    zeroshot: bool = True,
    options: list[str] | None = None,
    cuda_devices: str | None = None,
) -> int:
    sync_mertools_config(verbose=False)
    issues = verify_mertools_config()
    if issues:
        print("Config verification failed:", file=sys.stderr)
        for item in issues:
            print(f"  - {item}", file=sys.stderr)
        return 1

    bl = load_yaml("baseline.yaml")
    paths = get_paths()
    mertools_root = paths["mertools_root"]
    project_root = paths["project_root"]
    cfg = cfg_path or bl["train"]["human"]["cfg"]
    dataset = dataset or bl["inference"]["dataset"]

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.training.mertools_entry",
        "inference_hybird.py",
        f"--cfg-path={cfg}",
        f"--dataset={dataset}",
    ]
    if zeroshot:
        cmd.append("--zeroshot")

    merged_opts = _inference_options_from_baseline()
    if options:
        merged_opts.extend(options)
    if merged_opts:
        cmd.extend(["--options", *merged_opts])

    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    print("Running:", " ".join(cmd), f"(cwd={mertools_root})")
    return subprocess.call(cmd, cwd=str(mertools_root), env=env)


def find_reason_npz(mertools_root: Path | None = None) -> list[Path]:
    """列出推理产出的 reason npz（不含 -openset）。"""
    paths = get_paths() if mertools_root is None else {"mertools_root": mertools_root}
    root = Path(paths["mertools_root"])
    bl = load_yaml("baseline.yaml")
    base = bl["inference"]["base_root"]
    pattern = root / f"{base}-mer2026ov"
    if not pattern.exists():
        pattern = root / base.replace("output/", "output/")  # fallback
    candidates = list(root.glob(f"{base}*/*.npz"))
    return [p for p in candidates if not p.name.endswith("-openset.npz")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AffectGPT inference (inference_hybird.py)")
    parser.add_argument("--cfg-path", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no-zeroshot", action="store_true")
    parser.add_argument("--cuda", default=None)
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        help="Extra --options key=value (repeatable)",
    )
    args = parser.parse_args()

    code = run_inference(
        cfg_path=args.cfg_path,
        dataset=args.dataset,
        zeroshot=not args.no_zeroshot,
        options=args.option,
        cuda_devices=args.cuda,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
