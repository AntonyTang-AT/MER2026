"""选择性专家路由 — exp015 SER-Triple。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.evaluation.mertools_bridge import parse_openset_string
from src.inference.triple_union import merge_triple_union

ExpertKey = Literal["triple", "v3_union", "hybrid_v3e14"]
RouteStrategy = Literal["ser_rules", "ser_lr"]

E15_NOISE_LABELS = frozenset({"worried", "concerned", "confused"})
FEATURE_NAMES = (
    "len_rl",
    "len_e14",
    "len_e15",
    "jaccard_rl_e14",
    "jaccard_rl_e15",
    "jaccard_e14_e15",
    "jaccard_triple_v3union",
    "e15_noise_ratio",
    "rl_empty",
)


def _label_set(text: str) -> set[str]:
    return {x.lower().strip() for x in parse_openset_string(text) if str(x).strip() and x.lower() != "neutral"}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def _label_f1(pred: list[str], gt: list[str]) -> float:
    ps, gs = set(pred), set(gt)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    return 2 * len(ps & gs) / (len(ps) + len(gs))


def _norm_list(text: str) -> list[str]:
    return sorted(_label_set(text))


@dataclass
class SampleFeatures:
    name: str
    len_rl: int
    len_e14: int
    len_e15: int
    jaccard_rl_e14: float
    jaccard_rl_e15: float
    jaccard_e14_e15: float
    jaccard_triple_v3union: float
    e15_noise_ratio: float
    rl_empty: bool

    def to_vector(self) -> list[float]:
        return [
            float(self.len_rl),
            float(self.len_e14),
            float(self.len_e15),
            self.jaccard_rl_e14,
            self.jaccard_rl_e15,
            self.jaccard_e14_e15,
            self.jaccard_triple_v3union,
            self.e15_noise_ratio,
            1.0 if self.rl_empty else 0.0,
        ]


@dataclass
class RouterModel:
    strategy: RouteStrategy
    classes: list[ExpertKey]
    weights: list[list[float]]
    bias: list[float]
    confidence_threshold: float = 0.65
    default_expert: ExpertKey = "triple"
    max_switch_rate: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RouterModel:
        return cls(
            strategy=data["strategy"],
            classes=list(data["classes"]),
            weights=data["weights"],
            bias=data["bias"],
            confidence_threshold=float(data.get("confidence_threshold", 0.65)),
            default_expert=data.get("default_expert", "triple"),
            max_switch_rate=float(data.get("max_switch_rate", 0.1)),
        )


def _merge_one_sample(
    rl_text: str,
    e14_text: str,
    e15_text: str,
    *,
    max_labels: int = 8,
    include_rl: bool = True,
    include_e15: bool = True,
    mode: str = "fifo",
) -> set[str]:
    """单样本标签 merge，避免全量 dict 遍历。"""
    from src.inference.triple_union import E15_DEPRIORITIZE_LABELS, merge_triple_union

    one_rl = {"__s__": rl_text if include_rl else "[]"}
    one_e14 = {"__s__": e14_text}
    one_e15 = {"__s__": e15_text if include_e15 else "[]"}
    if not include_rl:
        merged = merge_triple_union({}, one_e14, one_e15, max_labels=max_labels, mode="fifo")
    elif mode == "e15_deprioritize":
        merged = merge_triple_union(
            one_rl, one_e14, one_e15, max_labels=max_labels, mode="e15_deprioritize"
        )
    else:
        merged = merge_triple_union(one_rl, one_e14, one_e15, max_labels=max_labels, mode="fifo")
    return _label_set(merged.get("__s__", ""))


def extract_features(
    name: str,
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
) -> SampleFeatures:
    s_rl = _label_set(rl.get(name, ""))
    s_e14 = _label_set(e14.get(name, ""))
    s_e15 = _label_set(e15.get(name, ""))
    triple_labels = _merge_one_sample(
        rl.get(name, ""), e14.get(name, ""), e15.get(name, ""), include_rl=True, include_e15=True
    )
    v3_labels = _merge_one_sample(
        rl.get(name, ""), e14.get(name, ""), e15.get(name, ""), include_rl=False, include_e15=True
    )
    e15_list = list(s_e15)
    noise_n = sum(1 for x in e15_list if x in E15_NOISE_LABELS)
    noise_ratio = noise_n / max(len(e15_list), 1)
    return SampleFeatures(
        name=name,
        len_rl=len(s_rl),
        len_e14=len(s_e14),
        len_e15=len(s_e15),
        jaccard_rl_e14=_jaccard(s_rl, s_e14),
        jaccard_rl_e15=_jaccard(s_rl, s_e15),
        jaccard_e14_e15=_jaccard(s_e14, s_e15),
        jaccard_triple_v3union=_jaccard(triple_labels, v3_labels),
        e15_noise_ratio=noise_ratio,
        rl_empty=len(s_rl) == 0,
    )


def build_expert_predictions(
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    *,
    max_labels: int = 8,
) -> dict[ExpertKey, dict[str, str]]:
    empty: dict[str, str] = {}
    return {
        "triple": merge_triple_union(rl, e14, e15, max_labels=max_labels, mode="fifo"),
        "v3_union": merge_triple_union(empty, e14, e15, max_labels=max_labels, mode="fifo"),
        "hybrid_v3e14": merge_triple_union(rl, e14, empty, max_labels=max_labels, mode="fifo"),
    }


def oracle_winner(
    name: str,
    experts: dict[ExpertKey, dict[str, str]],
    gt_labels: list[str],
) -> ExpertKey:
    best_key: ExpertKey = "triple"
    best_f1 = -1.0
    for key, preds in experts.items():
        pred_labels = _norm_list(preds.get(name, ""))
        f1 = _label_f1(pred_labels, gt_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_key = key
    return best_key


def route_rules(features: SampleFeatures, *, default_expert: ExpertKey = "triple") -> ExpertKey:
    """保守规则路由：默认 triple，仅极端情况切 v3_union。"""
    if features.rl_empty and features.e15_noise_ratio > 0.5:
        return "v3_union"
    return default_expert


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _train_logistic_ovr(
    X: list[list[float]],
    y: list[int],
    *,
    classes: list[ExpertKey],
    epochs: int = 800,
    lr: float = 0.05,
) -> tuple[list[list[float]], list[float]]:
    """One-vs-rest logistic regression（numpy 实现）。"""
    n_features = len(X[0])
    n_classes = len(classes)
    weights = [[0.0] * n_features for _ in range(n_classes)]
    bias = [0.0] * n_classes

    for cls_idx in range(n_classes):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                logits = sum(w * x for w, x in zip(weights[cls_idx], xi)) + bias[cls_idx]
                pred = _sigmoid(logits)
                target = 1.0 if yi == cls_idx else 0.0
                err = pred - target
                for j in range(n_features):
                    weights[cls_idx][j] -= lr * err * xi[j]
                bias[cls_idx] -= lr * err
    return weights, bias


def train_router_lr(
    names: list[str],
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    gt_labels: dict[str, list[str]],
    *,
    classes: list[ExpertKey] | None = None,
) -> RouterModel:
    classes = classes or ["triple", "v3_union"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    experts = build_expert_predictions(rl, e14, e15)

    X: list[list[float]] = []
    y: list[int] = []
    for name in names:
        feats = extract_features(name, rl, e14, e15)
        gt = gt_labels.get(name, [])
        winner = oracle_winner(name, experts, gt)
        if winner not in class_to_idx:
            winner = "triple"
        X.append(feats.to_vector())
        y.append(class_to_idx[winner])

    # 尝试 sklearn，失败则用 numpy
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        clf = LogisticRegression(max_iter=1000)
        clf.fit(np.array(X), np.array(y))
        weights = clf.coef_.tolist()
        bias = clf.intercept_.tolist()
        # 保持 classes 顺序与 sklearn 一致
        sklearn_classes = [classes[int(i)] for i in clf.classes_]
        classes = sklearn_classes
    except Exception:
        weights, bias = _train_logistic_ovr(X, y, classes=classes)

    return RouterModel(strategy="ser_lr", classes=classes, weights=weights, bias=bias)


def predict_lr(
    features: SampleFeatures,
    model: RouterModel,
) -> tuple[ExpertKey, float]:
    vec = features.to_vector()
    n_classes = len(model.classes)
    logits: list[float] = []
    if len(model.weights) == 1 and n_classes == 2:
        # sklearn binary logistic: single weight row
        logit = sum(w * x for w, x in zip(model.weights[0], vec)) + model.bias[0]
        prob_pos = _sigmoid(logit)
        probs = [1.0 - prob_pos, prob_pos]
    else:
        logits = [
            sum(w * x for w, x in zip(model.weights[i], vec)) + model.bias[i]
            for i in range(len(model.weights))
        ]
        probs = _softmax(logits)
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return model.classes[best_idx], probs[best_idx]


def route_sample(
    name: str,
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    *,
    strategy: RouteStrategy = "ser_rules",
    model: RouterModel | None = None,
    default_expert: ExpertKey = "triple",
    confidence_threshold: float = 0.65,
) -> ExpertKey:
    feats = extract_features(name, rl, e14, e15)
    if strategy == "ser_rules":
        return route_rules(feats, default_expert=default_expert)
    if strategy == "ser_lr" and model is not None:
        pred, conf = predict_lr(feats, model)
        if conf < confidence_threshold:
            return default_expert
        return pred
    return default_expert


def apply_routing(
    rl: dict[str, str],
    e14: dict[str, str],
    e15: dict[str, str],
    names: list[str],
    *,
    strategy: RouteStrategy = "ser_rules",
    model: RouterModel | None = None,
    max_labels: int = 8,
    default_expert: ExpertKey = "triple",
    confidence_threshold: float = 0.65,
    max_switch_rate: float | None = None,
) -> tuple[dict[str, str], dict[str, ExpertKey]]:
    experts = build_expert_predictions(rl, e14, e15, max_labels=max_labels)
    routes: dict[str, ExpertKey] = {}
    switch_names: list[str] = []

    for name in names:
        expert = route_sample(
            name, rl, e14, e15,
            strategy=strategy,
            model=model,
            default_expert=default_expert,
            confidence_threshold=confidence_threshold,
        )
        if expert != default_expert:
            switch_names.append(name)
        routes[name] = expert

    if max_switch_rate is not None and names:
        limit = int(len(names) * max_switch_rate)
        if len(switch_names) > limit:
            # 保留切换，其余回退 default
            keep = set(switch_names[:limit])
            for name in names:
                if routes[name] != default_expert and name not in keep:
                    routes[name] = default_expert

    out: dict[str, str] = {}
    for name in names:
        key = routes[name]
        out[name] = experts[key].get(name, "[]")
    return out, routes


def save_router_model(model: RouterModel, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_router_model(path: Path | str) -> RouterModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RouterModel.from_dict(data)


def switch_rate(routes: dict[str, ExpertKey], *, default_expert: ExpertKey = "triple") -> float:
    if not routes:
        return 0.0
    n_switch = sum(1 for v in routes.values() if v != default_expert)
    return n_switch / len(routes)
