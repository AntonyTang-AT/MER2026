"""批量路由入口 — 串联 3.1–3.4。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.core.config_loader import get_project_root, load_yaml
from src.core.types import RoutingResult
from src.data.dataset_index import DatasetSample, MER2026Index
from src.routing.expert_rules import classify_scores
from src.routing.modality_scorer import score_sample, valence_dict
from src.routing.va_distance import compute_distances, has_contradiction
from src.routing.weight_selector import select_weights


def route_sample(sample: DatasetSample) -> RoutingResult:
    scores = score_sample(sample)
    dist = compute_distances(scores)
    rule = classify_scores(scores)
    cfg = load_yaml("routing/contradiction_rules.yaml")
    threshold = float(cfg.get("distance", {}).get("contradiction_threshold", 0.6))

    ctype = rule.contradiction_type
    if ctype == "consistent" and has_contradiction(dist, threshold=threshold):
        ctype = "intensity_mismatch"

    weights, conf = select_weights(ctype, dist.max_distance)
    return RoutingResult(
        name=sample.name,
        contradiction_type=ctype,
        fusion_weights=weights,
        routing_confidence=conf,
        modality_scores=valence_dict(scores),
    )


def run_batch(
    samples: list[DatasetSample],
    *,
    output_json: Path | None = None,
    output_csv: Path | None = None,
) -> list[RoutingResult]:
    results = [route_sample(s) for s in samples]

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "name": r.name,
                "contradiction_type": r.contradiction_type,
                "fusion_weights": r.fusion_weights,
                "routing_confidence": r.routing_confidence,
                "modality_scores": r.modality_scores,
            }
            for r in results
        ]
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "name",
            "contradiction_type",
            "routing_confidence",
            "w_text",
            "w_audio",
            "w_face",
            "w_frame",
            "v_text",
            "v_audio",
            "v_face",
            "v_frame",
        ]
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow(
                    {
                        "name": r.name,
                        "contradiction_type": r.contradiction_type,
                        "routing_confidence": f"{r.routing_confidence:.4f}",
                        "w_text": f"{r.fusion_weights.get('text', 0):.4f}",
                        "w_audio": f"{r.fusion_weights.get('audio', 0):.4f}",
                        "w_face": f"{r.fusion_weights.get('face', 0):.4f}",
                        "w_frame": f"{r.fusion_weights.get('frame', 0):.4f}",
                        "v_text": f"{r.modality_scores.get('text', 0):.4f}",
                        "v_audio": f"{r.modality_scores.get('audio', 0):.4f}",
                        "v_face": f"{r.modality_scores.get('face', 0):.4f}",
                        "v_frame": f"{r.modality_scores.get('frame', 0):.4f}",
                    }
                )
    return results


def _summary(results: list[RoutingResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.contradiction_type] = counts.get(r.contradiction_type, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MER2026 L4 batch routing")
    parser.add_argument("--split", choices=["human", "mercaptionplus", "candidate"], default="human")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-media", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: outputs/routing/{split}_routing.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Default: outputs/routing/{split}_routing.csv",
    )
    args = parser.parse_args(argv)

    root = get_project_root()
    out_dir = root / "outputs" / "routing"
    json_path = args.output_json or out_dir / f"{args.split}_routing.json"
    csv_path = args.output_csv or out_dir / f"{args.split}_routing.csv"

    index = MER2026Index.from_config()
    samples = index.load_split(
        args.split,
        check_media=args.check_media,
        limit=args.limit,
    )
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        return 1

    results = run_batch(samples, output_json=json_path, output_csv=csv_path)
    summary = _summary(results)
    print(f"Routed {len(results)} samples -> {json_path}")
    print("Contradiction summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
