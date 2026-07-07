"""Qwen 开放词汇标签抽取 — 官方封装 + 自研 Prompt/后处理。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.core.config_loader import load_yaml
from src.evaluation.mertools_bridge import mertools_context
from src.inference.openset_postprocess import default_postprocess_config, postprocess_openset
from src.prompts.openset_builder import build_openset_prompt_batch
from src.training.mertools_entry import apply_compat_shims
from src.training.mertools_paths import get_paths, sync_mertools_config


def run_ovlabel_extraction(
    *,
    reason_npz: Path | None = None,
    store_npz: Path | None = None,
    cuda_devices: str | None = None,
    process_all: bool = False,
    prompt_variant: str = "official",
    postprocess_mode: str = "official",
) -> int:
    """调用 ovlabel 抽取逻辑。

    prompt_variant=official 且 postprocess_mode=official 时走官方脚本；
    否则使用自研 Prompt + 后处理。
    """
    if process_all and prompt_variant == "official" and postprocess_mode == "official":
        return _run_official_all(cuda_devices)

    if reason_npz is None or store_npz is None:
        if process_all:
            return _run_custom_all(
                cuda_devices=cuda_devices,
                prompt_variant=prompt_variant,
                postprocess_mode=postprocess_mode,
            )
        raise ValueError("reason_npz and store_npz required when process_all=False")

    return extract_openset_custom(
        reason_npz=reason_npz,
        store_npz=store_npz,
        cuda_devices=cuda_devices,
        prompt_variant=prompt_variant,
        postprocess_mode=postprocess_mode,
    )


def _run_official_all(cuda_devices: str | None) -> int:
    sync_mertools_config(verbose=False)
    paths = get_paths()
    mertools_root = paths["mertools_root"]
    project_root = paths["project_root"]
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.training.mertools_entry",
        "ovlabel_extraction.py",
    ]
    print("Running:", " ".join(cmd), f"(cwd={mertools_root})")
    return subprocess.call(cmd, cwd=str(mertools_root), env=env)


def _run_custom_all(
    *,
    cuda_devices: str | None,
    prompt_variant: str,
    postprocess_mode: str,
) -> int:
    sync_mertools_config(verbose=False)
    paths = get_paths()
    mertools_root = Path(paths["mertools_root"])
    bl = load_yaml("baseline.yaml")
    base_root = bl["inference"]["base_root"]
    pattern = str(mertools_root / f"{base_root}*/*.npz")
    import glob

    fail = 0
    for result_path in glob.glob(pattern):
        if result_path.endswith("-openset.npz"):
            continue
        openset_npz = result_path[:-4] + "-openset.npz"
        if os.path.exists(openset_npz):
            continue
        code = extract_openset_custom(
            reason_npz=Path(result_path),
            store_npz=Path(openset_npz),
            cuda_devices=cuda_devices,
            prompt_variant=prompt_variant,
            postprocess_mode=postprocess_mode,
        )
        if code != 0:
            fail = code
    return fail


def extract_openset_custom(
    *,
    reason_npz: Path,
    store_npz: Path,
    cuda_devices: str | None = None,
    prompt_variant: str = "ew_aware",
    postprocess_mode: str = "ew",
    batch_size: int = 8,
) -> int:
    """自研 Prompt + vllm batch + 后处理。"""
    apply_compat_shims()
    sync_mertools_config(verbose=False)

    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    name2reason = np.load(reason_npz, allow_pickle=True)["name2reason"].tolist()
    whole_names = list(name2reason.keys())
    pp_cfg = default_postprocess_config(mode=postprocess_mode) if postprocess_mode == "ew" else None

    with mertools_context():
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        import config as mconfig
        from toolkit.utils.functions import split_list_into_batch
        from toolkit.utils.qwen import func_postprocess_qwen, get_completion_qwen_bacth

        bl = load_yaml("baseline.yaml")
        modelname = bl["ovlabel"]["modelname"]
        model_path = mconfig.PATH_TO_LLM[modelname]

        if prompt_variant == "official" and postprocess_mode == "official":
            llm = LLM(model=model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            sampling_params = SamplingParams(
                temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512
            )
            whole_responses: list[str] = []
            for batch_names in split_list_into_batch(whole_names, batchsize=batch_size):
                batch_reasons = [name2reason[name] for name in batch_names]
                from toolkit.utils.qwen import reason_to_openset_qwen

                batch_responses = reason_to_openset_qwen(
                    llm=llm,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    batch_reasons=batch_reasons,
                )
                whole_responses.extend(batch_responses)
        else:
            llm = LLM(model=model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            sampling_params = SamplingParams(
                temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512
            )
            whole_responses = []
            for batch_names in split_list_into_batch(whole_names, batchsize=batch_size):
                batch_reasons = [name2reason[name] for name in batch_names]
                prompt_list = build_openset_prompt_batch(batch_reasons, variant=prompt_variant)
                batch_raw = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
                for raw in batch_raw:
                    if postprocess_mode == "ew":
                        whole_responses.append(postprocess_openset(raw, cfg=pp_cfg))
                    else:
                        whole_responses.append(func_postprocess_qwen(raw))

    store_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        store_npz,
        filenames=whole_names,
        fileitems=whole_responses,
    )
    print(f"Extract openset (custom): {reason_npz} -> {store_npz} (n={len(whole_names)})")
    return 0


def openset_path_for_reason(reason_npz: Path) -> Path:
    name = reason_npz.name
    if name.endswith(".npz"):
        return reason_npz.with_name(name[:-4] + "-openset.npz")
    return reason_npz.parent / f"{reason_npz.stem}-openset.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract openset labels via Qwen (ovlabel)")
    parser.add_argument("--reason-npz", type=Path, default=None, help="Single reason npz")
    parser.add_argument("--store-npz", type=Path, default=None, help="Output openset npz")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="process_all",
        help="Run batch ovlabel on all pending reason npz",
    )
    parser.add_argument("--cuda", default=None)
    parser.add_argument(
        "--prompt-variant",
        choices=["official", "ew_aware"],
        default="official",
        help="Openset prompt template variant",
    )
    parser.add_argument(
        "--postprocess",
        choices=["official", "ew"],
        default="official",
        help="Postprocess mode for raw Qwen output",
    )
    args = parser.parse_args()

    store = args.store_npz
    if args.reason_npz and store is None:
        store = openset_path_for_reason(args.reason_npz)

    code = run_ovlabel_extraction(
        reason_npz=args.reason_npz,
        store_npz=store,
        cuda_devices=args.cuda,
        process_all=args.process_all,
        prompt_variant=args.prompt_variant,
        postprocess_mode=args.postprocess,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
