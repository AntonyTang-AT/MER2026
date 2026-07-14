"""从 reason 文本补标签，提升 union 低 recall 样本的覆盖。"""

from __future__ import annotations

import re
from pathlib import Path

from src.evaluation.mertools_bridge import load_npz_predictions, parse_openset_string
from src.inference.openset_postprocess import (
    _load_known_labels_from_wheel,
    format_openset_list,
    sanitize_openset_string,
)


def _tokenize_reason(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return [t for t in text.split() if len(t) >= 3]


def _labels_from_reason(
    reason: str,
    known: set[str],
    *,
    exclude: set[str],
) -> list[str]:
    hits: list[str] = []
    seen = set(exclude)
    reason_l = reason.lower()
    for lab in sorted(known, key=len, reverse=True):
        key = lab.lower()
        if key in seen:
            continue
        if re.search(rf"\b{re.escape(key)}\b", reason_l):
            hits.append(lab)
            seen.add(key)
    if hits:
        return hits
    for tok in _tokenize_reason(reason):
        if tok in known and tok not in seen:
            hits.append(tok)
            seen.add(tok)
    return hits


def boost_predictions(
    preds: dict[str, str],
    reasons: dict[str, str],
    *,
    max_labels: int = 8,
    min_labels_for_boost: int = 4,
    max_add: int = 2,
) -> dict[str, str]:
    """对标签偏少的样本，从 reason 中补 1–2 个 EW 已知标签。"""
    known = _load_known_labels_from_wheel()
    if not known:
        return dict(preds)

    out: dict[str, str] = {}
    boosted_n = 0
    for name, pred_str in preds.items():
        labels = parse_openset_string(pred_str)
        if len(labels) >= min_labels_for_boost:
            out[name] = pred_str
            continue
        reason = reasons.get(name, "")
        if not reason.strip():
            out[name] = pred_str
            continue
        exclude = {lab.lower() for lab in labels}
        extra = _labels_from_reason(reason, known, exclude=exclude)[:max_add]
        if not extra:
            out[name] = pred_str
            continue
        merged = labels + extra
        merged = merged[:max_labels]
        sanitized = parse_openset_string(
            sanitize_openset_string("[" + ", ".join(merged) + "]")
        )
        final = sanitized[:max_labels]
        out[name] = format_openset_list(final) if final else pred_str
        if extra:
            boosted_n += 1
    return out


def load_reason_map(path: Path) -> dict[str, str]:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    if "name2reason" in data:
        raw = data["name2reason"].item()
        return {str(k): str(v) for k, v in raw.items()}
    raise ValueError(f"Unsupported reason npz: {path}")


def apply_recall_boost_file(
    pred_npz: Path,
    reason_npz: Path,
    out_npz: Path,
    **kwargs,
) -> dict[str, int]:
    preds = load_npz_predictions(pred_npz)
    reasons = load_reason_map(reason_npz)
    boosted = boost_predictions(preds, reasons, **kwargs)
    from src.inference.ensemble_runner import save_merged_predictions

    save_merged_predictions(boosted, out_npz)
    n_boosted = sum(1 for k in preds if preds.get(k) != boosted.get(k))
    return {"n_samples": len(boosted), "n_boosted": n_boosted}
