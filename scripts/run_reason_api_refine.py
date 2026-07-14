#!/usr/bin/env python3
"""用外部大模型 API 改写 reason npz，供后续 openset / RRB 使用。

示例::

  export REASON_API_KEY=sk-xxx
  export REASON_API_BASE_URL=https://api.deepseek.com
  export REASON_API_MODEL=deepseek-chat

  # 先 dry-run 看流程
  python scripts/run_reason_api_refine.py \\
    --in-reason <affectgpt_reason.npz> \\
    --out-reason outputs/reason_api/refined.npz \\
    --dry-run

  # 仅改写分歧子集
  python scripts/run_reason_api_refine.py \\
    --in-reason <affectgpt_reason.npz> \\
    --out-reason outputs/reason_api/refined_divergent.npz \\
    --names-json outputs/exp015/divergent_samples_candidate20k.json \\
    --limit 20

改写完成后，把 --out-reason 当作原 reason 路径，接 openset_extractor 或 RRB。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.reason_api import (
    load_names_json,
    load_reason_api_config,
    load_reason_map,
    save_reason_map,
)
from src.inference.reason_api.provider import open_provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine AffectGPT reason via external LLM API")
    parser.add_argument("--in-reason", type=Path, required=True, help="Input reason npz (name2reason)")
    parser.add_argument("--out-reason", type=Path, required=True, help="Output refined reason npz")
    parser.add_argument("--config", type=Path, default=None, help="config/reason_api.yaml")
    parser.add_argument("--names-json", type=Path, default=None, help="Optional subset names JSON")
    parser.add_argument("--limit", type=int, default=None, help="Only first N names (debug)")
    parser.add_argument("--provider", type=str, default=None, help="Override provider")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["refine", "generate_from_clues"])
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    cfg = load_reason_api_config(str(args.config) if args.config else None)
    if args.provider:
        cfg.provider = args.provider
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model
    if args.mode:
        cfg.mode = args.mode
    if args.concurrency is not None:
        cfg.max_concurrency = args.concurrency
    if args.dry_run:
        cfg.dry_run = True

    reasons = load_reason_map(args.in_reason)
    names = sorted(reasons.keys())
    if args.names_json:
        subset = set(load_names_json(args.names_json))
        names = [n for n in names if n in subset]
    if args.limit is not None:
        names = names[: max(0, args.limit)]

    print(
        f"reason_api provider={cfg.provider} model={cfg.model} mode={cfg.mode} "
        f"n={len(names)} dry_run={cfg.dry_run} base_url={cfg.base_url}",
        flush=True,
    )

    with open_provider(cfg) as provider:
        refined_part = provider.refine_map(reasons, names=names)

    # 全集：未选中的保持原样；选中的用改写结果
    out_map = dict(reasons)
    for n in names:
        out_map[n] = refined_part.get(n, reasons.get(n, ""))

    save_reason_map(out_map, args.out_reason)

    n_changed = sum(1 for n in names if out_map.get(n, "") != reasons.get(n, ""))
    report = {
        "in_reason": str(args.in_reason),
        "out_reason": str(args.out_reason),
        "provider": cfg.provider,
        "model": cfg.model,
        "mode": cfg.mode,
        "n_selected": len(names),
        "n_total": len(reasons),
        "n_changed": n_changed,
        "dry_run": cfg.dry_run,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
