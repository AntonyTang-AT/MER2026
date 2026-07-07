"""模态间 VA 欧氏距离 — L4.1。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.routing.modality_scorer import ModalityVA

MODALITIES = ("text", "audio", "face", "frame")


@dataclass(frozen=True)
class VADistanceResult:
    matrix: np.ndarray
    max_distance: float
    max_pair: tuple[str, str]

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "max_distance": self.max_distance,
            "max_pair": list(self.max_pair),
        }


def pairwise_distance_matrix(scores: dict[str, ModalityVA]) -> np.ndarray:
    """4×4 对称距离矩阵，对角线为 0。"""
    n = len(MODALITIES)
    mat = np.zeros((n, n), dtype=np.float64)
    for i, mi in enumerate(MODALITIES):
        for j, mj in enumerate(MODALITIES):
            if i >= j:
                continue
            vi, ai = scores[mi].valence, scores[mi].arousal
            vj, aj = scores[mj].valence, scores[mj].arousal
            d = float(np.sqrt((vi - vj) ** 2 + (ai - aj) ** 2))
            mat[i, j] = mat[j, i] = d
    return mat


def compute_distances(scores: dict[str, ModalityVA]) -> VADistanceResult:
    mat = pairwise_distance_matrix(scores)
    if mat.size == 0:
        return VADistanceResult(mat, 0.0, ("text", "audio"))

    idx = np.unravel_index(int(np.argmax(mat)), mat.shape)
    i, j = int(idx[0]), int(idx[1])
    pair = (MODALITIES[min(i, j)], MODALITIES[max(i, j)])
    return VADistanceResult(
        matrix=mat,
        max_distance=float(mat[i, j]),
        max_pair=pair,
    )


def has_contradiction(
    result: VADistanceResult,
    *,
    threshold: float = 0.6,
) -> bool:
    return result.max_distance >= threshold
