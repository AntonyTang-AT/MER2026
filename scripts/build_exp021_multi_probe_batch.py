#!/usr/bin/env python3
"""exp021 — 锁定 RRB-gap1 生产，换轴小改动率多探针批。

失败总结（exp020）：norm / gap1_norm / gap2 / SER0.60 均低于 65.7084%。
本批禁止扩大 RRB；仅做：收紧(gap1_pool)、cap、DTRB reason、保守 SER。
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.config_loader import get_project_root, load_global_config
from src.data.human_ov_split import load_val_names
from src.evaluation.ew_metric import compute_ew_f1
from src.evaluation.mertools_bridge import load_npz_predictions, parse_openset_string
from src.inference.dtrb_boost import DTRBConfig, apply_dtrb_boost
from src.inference.ensemble_runner import save_merged_predictions
from src.inference.expert_router import apply_routing, load_router_model
from src.inference.recall_boost import load_reason_map
from src.inference.rl_openset_bridge import BridgeConfig, bridge_rl_openset
from src.inference.triple_union import merge_triple_union

PROD_TEST = 0.657084
PROD_STACK_VAL = 0.6868086024388534

REASON_VAL = (
    "third_party/MERTools/MER2026/MER2026_Track2/output/"
    "results-mer2026ov-rlval-20260712171-human/"
    "human_outputhybird_bestsetup_bestfusion_face_lz_20260712171/"
    "checkpoint_000003_loss_0.128.npz"
)
REASON_C20K = (
    "third_party/MERTools/MER2026/MER2026_Track2/output/"
    "results-mer2026ov-rl-e3-mer2026ov/"
    "human_outputhybird_bestsetup_bestfusion_face_lz_20260712171/"
    "checkpoint_000003_exp014_rl_candidate.npz"
)
OFFICIAL_RL_VAL = (
    "third_party/MERTools/MER2026/MER2026_Track2/output/"
    "results-mer2026ov-rlval-20260712171-human/"
    "human_outputhybird_bestsetup_bestfusion_face_lz_20260712171/"
    "checkpoint_000003_loss_0.128-openset.npz"
)
E14_VAL = (
    "third_party/MERTools/MER2026/MER2026_Track2/output/results-mer2026ov-human/"
    "human_outputhybird_bestsetup_bestfusion_face_lz_20260712025/"
    "checkpoint_000014_loss_1.479-openset.npz"
)
E15_VAL = (
    "third_party/MERTools/MER2026/MER2026_Track2/output/results-mer2026ov-human/"
    "human_outputhybird_bestsetup_bestfusion_face_lz_20260712025/"
    "checkpoint_000015_loss_1.128-openset.npz"
)


def _labs(s: str) -> set[str]:
    return {x.lower().strip() for x in parse_openset_string(s) if str(x).strip()}


def _subset(raw: dict[str, str], names: list[str]) -> dict[str, str]:
    return {n: raw[n] for n in names if n in raw}


def _gap1(openset: dict[str, str], reasons: dict[str, str], names: list[str] | None = None) -> dict[str, str]:
    out, _ = bridge_rl_openset(
        openset, reasons,
        cfg=BridgeConfig(mode="gap_add", max_add=1, allow_noise_swap=True, require_gap=True),
        process_names=names,
    )
    return out


def _gap1_pool(
    openset: dict[str, str],
    reasons: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    names: list[str] | None = None,
) -> tuple[dict[str, str], int]:
    """gap1 收紧：仅当 recall 目标已在 e14∪e15 中才保留补标。"""
    gap = _gap1(openset, reasons, names=names)
    keys = names or sorted(gap.keys())
    out = dict(openset)
    kept = 0
    for n in keys:
        base = _labs(openset.get(n, ""))
        g = _labs(gap.get(n, ""))
        added = g - base
        if not added:
            out[n] = openset.get(n, gap.get(n, "[]"))
            continue
        pool = _labs(e14.get(n, "")) | _labs(e15.get(n, ""))
        allow = {t for t in added if t in pool}
        if not allow:
            out[n] = openset.get(n, "[]")
            continue
        # keep original + pool-gated adds only
        merged = list(base)
        for t in allow:
            if t not in merged:
                merged.append(t)
        from src.inference.openset_postprocess import format_openset_list

        out[n] = format_openset_list(merged[:8])
        kept += 1
    return out, kept


@dataclass
class Spec:
    tag: str
    priority: str
    kind: str  # gap1_pool | dtrb_reason | cap7 | ser70
    description: str


SPECS = [
    Spec("gap1_pool", "P0", "gap1_pool", "收紧gap1：仅保留与e14/e15对齐的补标"),
    Spec("dtrb_reason", "P0", "dtrb_reason", "生产gap1 RL + DTRB reason_guided"),
    Spec("cap7", "P1", "cap7", "生产组件 + triple/SER/DTRB cap7"),
    Spec("ser70_sw5", "P1", "ser70", "生产gap1 + SER conf=0.70 sw=5%"),
]


def _build_preds(
    rl, e14, e15, names, model, *,
    ser_conf=0.65, ser_sw=0.10, cap=8, dtrb_cfg=None, reasons=None,
):
    from src.inference.openset_postprocess import format_openset_list

    ser_preds, _ = apply_routing(
        rl, e14, e15, names,
        strategy="ser_lr", model=model,
        confidence_threshold=ser_conf, max_switch_rate=ser_sw,
    )
    if cap != 8:
        capped = {}
        for n, v in ser_preds.items():
            labs = list(parse_openset_string(v))[:cap]
            capped[n] = format_openset_list(labs)
        ser_preds = capped
    cfg = dtrb_cfg or DTRBConfig()
    if cap != 8:
        cfg = DTRBConfig(
            divergent_jaccard=cfg.divergent_jaccard,
            max_add=cfg.max_add,
            noise_swap_threshold=cfg.noise_swap_threshold,
            boost_on_noise=cfg.boost_on_noise,
            max_labels=cap,
            reason_guided=bool(getattr(cfg, "reason_guided", False)),
        )
    dtrb_preds, added = apply_dtrb_boost(
        ser_preds, rl, e14, e15, cfg=cfg, names=names, reasons=reasons,
    )
    return dtrb_preds, len(added)


def _stack(
    rl, e14, e15, names, gt_csv, model, *,
    ser_conf=0.65, ser_sw=0.10, cap=8, dtrb_cfg=None, reasons=None,
):
    triple = merge_triple_union(rl, e14, e15, max_labels=cap, mode="fifo")
    dtrb_preds, n_boosted = _build_preds(
        rl, e14, e15, names, model,
        ser_conf=ser_conf, ser_sw=ser_sw, cap=cap, dtrb_cfg=dtrb_cfg, reasons=reasons,
    )
    return {
        "triple": compute_ew_f1(triple, gt_csv=gt_csv, process_names=names).ew_f1,
        "stack": compute_ew_f1(dtrb_preds, gt_csv=gt_csv, process_names=names).ew_f1,
        "n_boosted": n_boosted,
        "preds": dtrb_preds,
    }


def _pack(root: Path, tag: str, preds: dict[str, str]) -> dict:
    out_dir = root / "outputs/exp021"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{tag}_candidate20k.npz"
    save_merged_predictions(preds, npz)
    csv = root / "outputs/submissions" / f"{tag}_candidate20k.csv"
    zipf = root / "outputs/submissions" / f"{tag}_candidate20k.zip"
    qa = root / "experiments/exp021_post_fail_batch" / f"qa_{tag}.json"
    qa.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python3", "scripts/qa_candidate_submission.py", str(npz), "--json-out", str(qa)],
        check=False, cwd=str(root),
    )
    subprocess.run(
        ["python3", "-m", "src.data.submission_formatter", "--pred", str(npz), "--out", str(csv)],
        check=True, cwd=str(root),
    )
    subprocess.run(
        f"cd {root}/outputs/submissions && cp {csv.name} answer.csv && zip -j {zipf.name} answer.csv && rm -f answer.csv",
        shell=True, check=True,
    )
    return {"npz": str(npz), "zip": str(zipf), "qa": str(qa)}


def main() -> None:
    root = get_project_root()
    out_dir = root / "outputs/exp021"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = load_val_names()
    gt_csv = root / load_global_config()["paths"]["data_root"] / "track2_train_human.csv"
    model = load_router_model(root / "outputs/exp015/ser_router_model.json")

    official_val = _subset(load_npz_predictions(root / OFFICIAL_RL_VAL), names)
    reason_val = load_reason_map(root / REASON_VAL)
    e14_val = _subset(load_npz_predictions(root / E14_VAL), names)
    e15_val = _subset(load_npz_predictions(root / E15_VAL), names)

    rl_gap1_c20k = load_npz_predictions(root / "outputs/exp019/RL_v2_e3_reason_bridge_gap1_candidate20k.npz")
    e14_c20k = load_npz_predictions(root / "outputs/exp014/SFT_v3_e14_candidate20k.npz")
    e15_c20k = load_npz_predictions(root / "outputs/exp014/SFT_v3_e15_candidate20k.npz")
    reason_c20k = load_reason_map(root / REASON_C20K)
    official_c20k = load_npz_predictions(root / "outputs/exp014/RL_v2_e3_candidate20k.npz")
    c20k_names = sorted(set(rl_gap1_c20k) & set(e14_c20k) & set(e15_c20k))

    # baseline production val (gap1)
    rl_gap1_val = _gap1(official_val, reason_val, names=names)
    prod_val_preds, _ = _build_preds(rl_gap1_val, e14_val, e15_val, names, model)

    results = []
    for spec in SPECS:
        print(f"=== {spec.tag} ({spec.priority}) ===", flush=True)
        ser_conf, ser_sw, cap = 0.65, 0.10, 8
        dtrb_cfg = DTRBConfig()
        reasons_val = reasons_c20k = None
        meta = {}

        if spec.kind == "gap1_pool":
            rl_val, n_kept = _gap1_pool(official_val, reason_val, e14_val, e15_val, names=names)
            rl_c20k, n_c20k = _gap1_pool(official_c20k, reason_c20k, e14_c20k, e15_c20k)
            meta = {"n_val_kept": n_kept, "n_c20k_kept": n_c20k}
        elif spec.kind == "dtrb_reason":
            rl_val = rl_gap1_val
            rl_c20k = rl_gap1_c20k
            dtrb_cfg = DTRBConfig(reason_guided=True)
            reasons_val, reasons_c20k = reason_val, reason_c20k
        elif spec.kind == "cap7":
            rl_val = rl_gap1_val
            rl_c20k = rl_gap1_c20k
            cap = 7
            dtrb_cfg = DTRBConfig(max_labels=7)
        elif spec.kind == "ser70":
            rl_val = rl_gap1_val
            rl_c20k = rl_gap1_c20k
            ser_conf, ser_sw = 0.70, 0.05
        else:
            raise ValueError(spec.kind)

        save_merged_predictions(rl_c20k, out_dir / f"RL_{spec.tag}_candidate20k.npz")

        m_val = _stack(
            rl_val, e14_val, e15_val, names, gt_csv, model,
            ser_conf=ser_conf, ser_sw=ser_sw, cap=cap,
            dtrb_cfg=dtrb_cfg, reasons=reasons_val,
        )
        chg = sum(1 for n in names if m_val["preds"].get(n) != prod_val_preds.get(n)) / max(len(names), 1)

        preds_c20k, n_boost = _build_preds(
            rl_c20k, e14_c20k, e15_c20k, c20k_names, model,
            ser_conf=ser_conf, ser_sw=ser_sw, cap=cap,
            dtrb_cfg=dtrb_cfg, reasons=reasons_c20k,
        )

        # 相对生产：允许微跌打包（多探针策略）；禁止显著回退
        pack_ok = m_val["stack"] >= PROD_STACK_VAL - 0.004 and chg > 0
        full_tag = f"R3_RL_triple_ser_lr_dtrb_{spec.tag}_cap{cap}"
        if abs(ser_conf - 0.65) > 1e-9:
            full_tag = f"R3_RL_triple_ser{str(ser_conf).replace('.','')}_sw{int(ser_sw*100)}_dtrb_{spec.tag}_cap{cap}"

        row = {
            "tag": full_tag,
            "short": spec.tag,
            "priority": spec.priority,
            "description": spec.description,
            "val_stack_ew_f1": m_val["stack"],
            "delta_pp_vs_prod": (m_val["stack"] - PROD_STACK_VAL) * 100,
            "val_change_rate_vs_prod": chg,
            "ser_conf": ser_conf,
            "ser_sw": ser_sw,
            "cap": cap,
            "meta": meta,
            "n_dtrb_boosted_c20k": n_boost,
            "pack_ok": pack_ok,
        }
        if pack_ok:
            paths = _pack(root, full_tag, preds_c20k)
            row.update(paths)
            print(
                f"  PACKED stack={m_val['stack']*100:.2f}% delta={row['delta_pp_vs_prod']:+.2f}pp "
                f"chg={chg*100:.1f}%",
                flush=True,
            )
        else:
            print(
                f"  SKIP stack={m_val['stack']*100:.2f}% chg={chg*100:.1f}%",
                flush=True,
            )
        results.append(row)

    packed = [r for r in results if r.get("pack_ok")]
    packed.sort(key=lambda r: (0 if r["priority"] == "P0" else 1, -r["val_stack_ew_f1"]))
    manifest = {
        "experiment": "exp021",
        "archived_exp020": {
            "ser06": 0.655347,
            "gap1_norm": 0.656094,
            "gap2": 0.656209,
            "norm": 0.656048,
            "production": PROD_TEST,
            "conclusion": "扩大RRB/加SER0.60全部回退；锁定gap1生产",
        },
        "production": {
            "variant": "R3_RL_triple_ser_lr_dtrb_rrb_reason_bridge_gap1_cap8",
            "test": PROD_TEST,
            "val_stack": PROD_STACK_VAL,
        },
        "probes": results,
        "submit_batch": [
            {"priority": r["priority"], "zip": r["zip"], "val_stack": r["val_stack_ew_f1"], "desc": r["description"]}
            for r in packed
        ],
        "n_packed": len(packed),
    }
    out = out_dir / "exp021_multi_probe_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== batch summary ===")
    for r in packed:
        print(f"  [{r['priority']}] {r['tag']} stack={r['val_stack_ew_f1']*100:.2f}% -> {r['zip']}")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
