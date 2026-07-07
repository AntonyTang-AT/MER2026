"""带 per-sample Prompt 的 AffectGPT 推理 — 不修改 third_party。"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

from src.core.config_loader import load_yaml
from src.inference.human_inference_hook import _patch_human_dataset_cls
from src.data.human_ov_split import val_split_path
from src.evaluation.mertools_bridge import mertools_context
from src.prompts.description_builder import build_description_prompt, load_routing_map
from src.training.mertools_entry import apply_compat_shims, apply_torch_amp_compat
from src.training.mertools_paths import get_paths, sync_mertools_config, verify_mertools_config


def _import_inference_helpers():
    """在 MERTools 上下文中导入官方 inference 依赖。"""
    apply_compat_shims()
    import my_affectgpt.datasets.builders  # noqa: F401
    import my_affectgpt.models  # noqa: F401
    import my_affectgpt.processors  # noqa: F401
    import my_affectgpt.runners  # noqa: F401
    import my_affectgpt.tasks  # noqa: F401

    from my_affectgpt.common.config import Config
    from my_affectgpt.common.registry import registry
    from my_affectgpt.conversation.conversation_video import Chat
    from my_affectgpt.datasets.builders.image_text_pair_builder import MER2026OV_Dataset
    from my_affectgpt.processors.base_processor import BaseProcessor

    import inference_hybird as inf

    return Config, registry, Chat, BaseProcessor, MER2026OV_Dataset, inf


def run_inference_with_prompts(
    *,
    cfg_path: str | None = None,
    dataset: str = "MER2026OV",
    routing_json: Path | None = None,
    prompt_variant: str = "routing",
    cuda_devices: str | None = None,
    limit: int | None = None,
    test_epoch: int | None = None,
    ckpt_root: str | Path | None = None,
) -> int:
    """逐样本注入 routing-aware user_message 并运行 AffectGPT 推理。"""
    apply_compat_shims()
    apply_torch_amp_compat()
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
    cfg_path = cfg_path or bl["train"]["human"]["cfg"]

    if routing_json is None:
        default_routing = project_root / "outputs" / "routing" / "human_routing.json"
        routing_json = default_routing if default_routing.is_file() else None

    name2routing = load_routing_map(routing_json) if routing_json else {}
    if prompt_variant == "routing" and not name2routing:
        print(
            f"WARN: prompt_variant=routing but no routing map at {routing_json}; "
            "falling back to default template.",
            file=sys.stderr,
        )

    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    if dataset == "Human":
        env["TMX_HUMAN_VAL_LIST"] = str(val_split_path())
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    infer_options = [
        f"inference.face_or_frame={bl['inference']['face_or_frame']}",
        f"inference.base_root={bl['inference']['base_root']}",
        f"inference.test_epochs={bl['inference']['test_epochs']}",
        f"inference.skip_epoch={bl['inference']['skip_epoch']}",
        f"inference.gpu={bl['inference']['gpu']}",
    ]
    if test_epoch is not None:
        infer_options.append(f"inference.test_epoch={test_epoch}")
    if ckpt_root is not None:
        infer_options.append(f"inference.ckpt_root={Path(ckpt_root).as_posix()}")

    with mertools_context():
        os.environ.update({k: v for k, v in env.items() if k != "PYTHONPATH"})
        Config, registry, Chat, BaseProcessor, MER2026OV_Dataset, inf = _import_inference_helpers()

        from types import SimpleNamespace

        args = SimpleNamespace(
            cfg_path=cfg_path,
            options=infer_options,
            dataset=dataset,
            zeroshot=True,
            outside_user_message=None,
            outside_face_or_frame=bl["inference"]["face_or_frame"],
        )
        cfg = Config(args)
        model_cfg = cfg.model_cfg
        datasets_cfg = cfg.datasets_cfg
        inference_cfg = cfg.inference_cfg
        device = f"cuda:{inference_cfg.gpu}"

        if inference_cfg.ckpt_root not in ["", "xxx"]:
            ckpt3_root = inference_cfg.ckpt_root
        elif inference_cfg.ckpt_name not in ["", "xxx"]:
            cfg_name = os.path.basename(args.cfg_path)[:-5]
            ckpt3_root = os.path.join("output", cfg_name, inference_cfg.ckpt_name)
        else:
            cfg_name = os.path.basename(args.cfg_path)[:-5]
            root_candidates = glob.glob(os.path.join("output", cfg_name, cfg_name + "*"))
            ckpt3_root = inf.search_for_ckpt_root(root_candidates)

        face_or_frame = inf.get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
        whole_ckpt3s = inf.get_ckpt3_candidates(ckpt3_root, inference_cfg)

        def get_name2cls(name: str):
            if name == "MER2026OV":
                return MER2026OV_Dataset()
            if name == "Human":
                _patch_human_dataset_cls()
                from my_affectgpt.datasets.datasets.human_dataset import Human_Dataset

                return Human_Dataset()
            raise ValueError(f"Unsupported dataset: {name}")

        for ckpt_idx, ckpt_3 in enumerate(whole_ckpt3s):
            print(f"======== ckpt: {os.path.basename(ckpt_3)} ========")
            model_cfg.ckpt_3 = ckpt_3
            if ckpt_idx == 0:
                model_cls = registry.get_model_class(model_cfg.arch)
                model = model_cls.from_config(model_cfg)
            else:
                ckpt = torch.load(model_cfg.ckpt_3, map_location="cpu", weights_only=True)
                model.load_state_dict(ckpt["model"], strict=False)
            model = model.to(device).eval()
            chat = Chat(model, model_cfg, device=device)

            dataset_cls = get_name2cls(dataset)
            dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
            dataset_cls.vis_processor = BaseProcessor()
            dataset_cls.img_processor = BaseProcessor()
            vis_processor_cfg = inference_cfg.get("vis_processor")
            img_processor_cfg = inference_cfg.get("img_processor")
            if vis_processor_cfg is not None:
                dataset_cls.vis_processor = registry.get_processor_class(
                    vis_processor_cfg.train.name
                ).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                dataset_cls.img_processor = registry.get_processor_class(
                    img_processor_cfg.train.name
                ).from_config(img_processor_cfg.train)
            dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms

            test_names = dataset_cls.read_test_names()
            if limit is not None:
                test_names = test_names[:limit]
            name2subtitle = dataset_cls.name2subtitle

            save_root = os.path.join(
                inference_cfg.base_root + f"-{dataset.lower()}-prompts",
                os.path.basename(ckpt3_root),
            )
            os.makedirs(save_root, exist_ok=True)
            epoch = os.path.basename(model_cfg.ckpt_3)[:-4]
            save_path = os.path.join(save_root, f"{epoch}.npz")
            if os.path.exists(save_path):
                print(f"Skip existing: {save_path}")
                continue

            name2reason: dict[str, str] = {}
            for ii, name in enumerate(test_names):
                subtitle = name2subtitle[name]
                routing = name2routing.get(name)
                effective_variant = prompt_variant
                if effective_variant == "routing" and routing is None:
                    effective_variant = "default"

                user_message = build_description_prompt(
                    subtitle=subtitle,
                    routing=routing,
                    variant=effective_variant,
                )
                print(f"process {ii + 1}/{len(test_names)}: {name}")

                sample = {"name": name}
                video_path = dataset_cls._get_video_path(sample) if hasattr(dataset_cls, "_get_video_path") else None
                audio_path = dataset_cls._get_audio_path(sample) if hasattr(dataset_cls, "_get_audio_path") else None
                face_npy = dataset_cls._get_face_path(sample) if hasattr(dataset_cls, "_get_face_path") else None
                image_path = dataset_cls._get_image_path(sample) if hasattr(dataset_cls, "_get_image_path") else None
                sample_data = dataset_cls.read_frame_face_audio_text(
                    video_path, face_npy, audio_path, image_path
                )

                audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
                frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
                face_hiddens, face_llms = chat.postprocess_face(sample_data)
                _, image_llms = chat.postprocess_image(sample_data)
                multi_llms = None
                if face_or_frame.startswith("multiface"):
                    _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
                elif face_or_frame.startswith("multiframe"):
                    _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)

                img_list = {
                    "audio": audio_llms,
                    "frame": frame_llms,
                    "face": face_llms,
                    "image": image_llms,
                    "multi": multi_llms,
                }
                prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)
                response = chat.answer_sample(
                    prompt=prompt,
                    img_list=img_list,
                    num_beams=1,
                    temperature=1,
                    do_sample=True,
                    top_p=0.9,
                    max_new_tokens=1200,
                    max_length=2000,
                )
                name2reason[name] = response

            np.savez_compressed(save_path, name2reason=name2reason)
            print(f"Saved: {save_path}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="AffectGPT inference with per-sample prompts")
    parser.add_argument("--cfg-path", default=None)
    parser.add_argument("--dataset", default="MER2026OV")
    parser.add_argument(
        "--routing-json",
        type=Path,
        default=None,
        help="Routing JSON (default: outputs/routing/human_routing.json)",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=["official", "default", "routing"],
        default="routing",
    )
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--test-epoch", type=int, default=None, help="Single checkpoint epoch")
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=None,
        help="Explicit checkpoint run directory (avoids auto-pick by file count)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: limit sample count")
    args = parser.parse_args()

    code = run_inference_with_prompts(
        cfg_path=args.cfg_path,
        dataset=args.dataset,
        routing_json=args.routing_json,
        prompt_variant=args.prompt_variant,
        cuda_devices=args.cuda,
        limit=args.limit,
        test_epoch=args.test_epoch,
        ckpt_root=args.ckpt_root,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
