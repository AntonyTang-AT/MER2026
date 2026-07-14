"""exp009 Tier1 ensemble — E1/E2/E3 融合评测。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.core.config_loader import get_project_root
from src.data.human_ov_split import load_val_names
from src.evaluation.mertools_bridge import load_npz_predictions, parse_openset_string
from src.inference.agent_mer_pipeline import _save_openset_npz, agent_mer_v2_dir, evaluate_openset_npz
from src.inference.ensemble import merge_openset_predictions
from src.inference.openset_postprocess import (
    PostprocessConfig,
    default_postprocess_config,
    format_openset_list,
    normalize_labels,
)
from src.inference.self_consistent_voter import vote_label_sets
from src.fusion.faithfulness_gate import FaithfulnessConfig, faithful_merge_labels
from src.inference.openset_postprocess import (
    Tier2PostprocessConfig,
    postprocess_labels_tier2,
)


def _cap_labels(labels: list[str], *, max_labels: int = 8) -> list[str]:
    if len(labels) <= max_labels:
        return labels
    return labels[:max_labels]


def _labels_to_openset(labels: list[str], *, cfg: PostprocessConfig | None = None) -> str:
    cfg = cfg or default_postprocess_config(mode="ew")
    normalized = normalize_labels(labels, cfg=cfg)
    return format_openset_list(normalized)


def _load_b7_path_plan() -> dict[str, str]:
    meta_path = agent_mer_v2_dir() / "B7_official_augment_v53_deepseek_agent_mindiv080_anchor.meta.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return dict(meta.get("paths") or {})


def build_e1_union_b7_c0(
    b7_npz: Path,
    c0_npz: Path,
    *,
    max_labels: int = 8,
) -> dict[str, str]:
    """E1: union(B7, C0) + EW 后处理 + max 8 labels。"""
    b7 = load_npz_predictions(b7_npz)
    c0 = load_npz_predictions(c0_npz)
    merged = merge_openset_predictions(
        [b7, c0],
        strategy="label_union",
        postprocess_cfg=default_postprocess_config(mode="ew"),
    )
    cfg = default_postprocess_config(mode="ew")
    out: dict[str, str] = {}
    for name, raw in merged.items():
        labels = _cap_labels(parse_openset_string(raw), max_labels=max_labels)
        out[name] = _labels_to_openset(labels, cfg=cfg)
    return out


def build_e2_vote_b7_c0_c1(
    b7_npz: Path,
    c0_npz: Path,
    c1_npz: Path,
    *,
    vote_k: int = 2,
) -> dict[str, str]:
    """E2: vote(B7, C0, C1), k=2。"""
    preds = [load_npz_predictions(p) for p in (b7_npz, c0_npz, c1_npz)]
    names = sorted(set().union(*(p.keys() for p in preds)))
    cfg = default_postprocess_config(mode="ew")
    out: dict[str, str] = {}
    for name in names:
        label_sets = [parse_openset_string(p.get(name, "[]")) for p in preds]
        voted = vote_label_sets(label_sets, k=vote_k)
        out[name] = _labels_to_openset(voted, cfg=cfg)
    return out


def build_e3_agent_union_b7_c0(
    b7_npz: Path,
    c0_npz: Path,
    *,
    max_labels: int = 8,
) -> dict[str, str]:
    """E3: B7 路由不变；Agent 路 union(B7, C0)（div≥0.80 → agent 样本）。"""
    b7 = load_npz_predictions(b7_npz)
    c0 = load_npz_predictions(c0_npz)
    path_plan = _load_b7_path_plan()
    cfg = default_postprocess_config(mode="ew")
    out: dict[str, str] = {}
    for name in sorted(set(b7.keys()) | set(c0.keys())):
        path = path_plan.get(name, "fast")
        if path == "agent":
            labels = parse_openset_string(b7.get(name, "[]")) + parse_openset_string(
                c0.get(name, "[]")
            )
            labels = _cap_labels(
                normalize_labels(labels, cfg=cfg),
                max_labels=max_labels,
            )
            out[name] = _labels_to_openset(labels, cfg=cfg)
        else:
            out[name] = b7.get(name, "[]")
    return out


def build_e4_faithful_union_b7_c0(
    b7_npz: Path,
    c0_npz: Path,
    *,
    ctx_map: dict[str, dict] | None = None,
    pp_cfg: Tier2PostprocessConfig | None = None,
) -> dict[str, str]:
    """E4: faithful_union(B7,C0) + Tier2 后处理。"""
    from src.inference.openset_postprocess import tier2_config_from_yaml

    b7 = load_npz_predictions(b7_npz)
    c0 = load_npz_predictions(c0_npz)
    cfg = FaithfulnessConfig.from_yaml()
    pp_cfg = pp_cfg or tier2_config_from_yaml()
    # 忠实度融合默认关闭 L1 dedup，避免相对 E1 过度削减 recall
    pp_cfg = Tier2PostprocessConfig(
        lowercase=pp_cfg.lowercase,
        deduplicate=pp_cfg.deduplicate,
        apply_synonym_map=pp_cfg.apply_synonym_map,
        filter_unknown=pp_cfg.filter_unknown,
        synonym_map=pp_cfg.synonym_map,
        known_labels=pp_cfg.known_labels,
        l1_dedup=False,
        max_per_l1=pp_cfg.max_per_l1,
        max_labels=pp_cfg.max_labels,
        adaptive_cap=False,
    )
    out: dict[str, str] = {}
    for name in sorted(set(b7.keys()) | set(c0.keys())):
        b7_labels = parse_openset_string(b7.get(name, "[]"))
        c0_labels = parse_openset_string(c0.get(name, "[]"))
        div = None
        if ctx_map and name in ctx_map:
            div = float(ctx_map[name].get("divergence_score", 0.0))
        merged = faithful_merge_labels(
            b7_labels, c0_labels, cfg=cfg, divergence=div
        )
        final = postprocess_labels_tier2(merged, cfg=pp_cfg)
        out[name] = _labels_to_openset(final, cfg=pp_cfg)
    return out


def build_e5_faithful_with_b8_agent(
    e4_preds: dict[str, str],
    b8_npz: Path,
    *,
    pp_cfg: Tier2PostprocessConfig | None = None,
) -> dict[str, str]:
    """E5: E4 fast 路 + B8 agent 路替换。"""
    from src.inference.openset_postprocess import tier2_config_from_yaml

    b8 = load_npz_predictions(b8_npz)
    path_plan = _load_b7_path_plan()
    pp_cfg = pp_cfg or tier2_config_from_yaml()
    out: dict[str, str] = {}
    for name in sorted(e4_preds.keys()):
        if path_plan.get(name, "fast") == "agent" and name in b8:
            labels = parse_openset_string(b8[name])
            final = postprocess_labels_tier2(labels, cfg=pp_cfg)
            out[name] = _labels_to_openset(final, cfg=pp_cfg)
        else:
            out[name] = e4_preds[name]
    return out


def save_merged_predictions(name2openset: dict[str, str], out_npz: Path) -> Path:
    names = list(name2openset.keys())
    items = [name2openset[n] for n in names]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, filenames=names, fileitems=items)
    return out_npz


def run_exp009_ensemble(
    *,
    b7_npz: Path | None = None,
    c0_npz: Path | None = None,
    c1_npz: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """构建 E1/E2/E3 npz 并返回路径映射。"""
    v2 = agent_mer_v2_dir()
    b7_npz = b7_npz or (v2 / "B7_official_augment_v53_deepseek_agent_mindiv080_anchor.npz")
    c0_npz = c0_npz or (v2 / "C0_qwen25omni_val.npz")
    c1_npz = c1_npz or (v2 / "C1_emotion_llama_val.npz")
    out_dir = out_dir or v2

    results: dict[str, Path] = {}

    e1 = build_e1_union_b7_c0(b7_npz, c0_npz)
    e1_path = out_dir / "E1_union_B7_C0.npz"
    save_merged_predictions(e1, e1_path)
    results["E1"] = e1_path

    if c0_npz.is_file():
        e3 = build_e3_agent_union_b7_c0(b7_npz, c0_npz)
        e3_path = out_dir / "E3_agent_union_B7_C0.npz"
        save_merged_predictions(e3, e3_path)
        results["E3"] = e3_path

    if c1_npz.is_file():
        e2 = build_e2_vote_b7_c0_c1(b7_npz, c0_npz, c1_npz)
        e2_path = out_dir / "E2_vote_B7_C0_C1.npz"
        save_merged_predictions(e2, e2_path)
        results["E2"] = e2_path

    return results


def evaluate_exp009_matrix(
    *,
    b7_npz: Path | None = None,
    c0_npz: Path | None = None,
    c1_npz: Path | None = None,
) -> Path:
    """评测 C0/C1/E1/E2/E3 vs B7，写入 experiments/exp009_tier1_sota/summary.csv。"""
    import csv

    v2 = agent_mer_v2_dir()
    b7_npz = b7_npz or (v2 / "B7_official_augment_v53_deepseek_agent_mindiv080_anchor.npz")
    c0_npz = c0_npz or (v2 / "C0_qwen25omni_val.npz")
    c1_npz = c1_npz or (v2 / "C1_emotion_llama_val.npz")

    ensemble_paths = run_exp009_ensemble(b7_npz=b7_npz, c0_npz=c0_npz, c1_npz=c1_npz)

    out_csv = get_project_root() / "experiments" / "exp009_tier1_sota" / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    candidates: list[tuple[str, str, Path]] = [
        ("B7", "gated+deepseek_agent_mindiv080_anchor", b7_npz),
        ("C0", "qwen25_omni_audiovideotext_ew", c0_npz),
        ("C1", "emotion_llama_ov_openset", c1_npz),
    ]
    for vid, mode, npz in candidates:
        if not npz.is_file():
            continue
        metrics = evaluate_openset_npz(npz)
        rows.append({"variant": vid, "mode": mode, **metrics})

    mode_map = {
        "E1": "union(B7,C0)+ew+max8",
        "E2": "vote(B7,C0,C1,k=2)",
        "E3": "B7_route+agent_union(B7,C0)",
    }
    for vid, npz in ensemble_paths.items():
        metrics = evaluate_openset_npz(npz)
        rows.append({"variant": vid, "mode": mode_map.get(vid, vid), **metrics})

    fieldnames = ["variant", "mode", "ew_f1", "precision", "recall"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("exp009 val results:")
    for r in rows:
        print(f"  {r['variant']}: EW-F1={float(r['ew_f1']):.4f}")
    if len(rows) > 1:
        best = max(rows, key=lambda r: float(r["ew_f1"]))
        print(f"  → Best: {best['variant']} ({float(best['ew_f1'])*100:.2f}%)")
    print(f"Summary written: {out_csv}")
    return out_csv
