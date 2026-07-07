"""各模态 valence/arousal 代理分数 — 规则版 L4 输入（无需 GPU）。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.data.dataset_index import DatasetSample

POSITIVE_WORDS = frozenset(
    "happy joy love glad pleased wonderful great good nice sweet warm positive excited".split()
)
NEGATIVE_WORDS = frozenset(
    "sad angry hate fear awful terrible bad wrong cry pain negative depressed anxious worried".split()
)
HIGH_AROUSAL_WORDS = frozenset(
    "! ? angry shout scream panic furious excited alarm urgent".split()
)


@dataclass(frozen=True)
class ModalityVA:
    """单模态 VA 代理与置信度。"""

    valence: float
    arousal: float
    confidence: float = 0.5

    def clamp(self) -> ModalityVA:
        return ModalityVA(
            valence=max(-1.0, min(1.0, self.valence)),
            arousal=max(-1.0, min(1.0, self.arousal)),
            confidence=max(0.0, min(1.0, self.confidence)),
        )


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_text(subtitle: str, *, openset: list[str] | None = None) -> ModalityVA:
    """字幕英文 lexicon + 可选 openset 标签微调。"""
    text = (subtitle or "").lower()
    tokens = re.findall(r"[a-z']+", text)
    if not tokens:
        return ModalityVA(0.0, 0.0, 0.2).clamp()

    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        valence = 0.0
    else:
        valence = _clip((pos - neg) / max(total, 1))

    arousal = _clip(
        0.3 * text.count("!")
        + 0.2 * text.count("?")
        + 0.1 * sum(1 for t in tokens if t in HIGH_AROUSAL_WORDS)
    )
    conf = _clip(0.35 + 0.1 * len(tokens), 0.0, 1.0)

    if openset:
        pos_labels = sum(
            1 for w in openset if any(p in w for p in ("happy", "joy", "warm", "positive"))
        )
        neg_labels = sum(
            1 for w in openset if any(n in w for n in ("sad", "angry", "fear", "negative"))
        )
        if pos_labels + neg_labels > 0:
            label_v = _clip((pos_labels - neg_labels) / (pos_labels + neg_labels))
            valence = _clip(0.6 * valence + 0.4 * label_v)
            conf = _clip(conf + 0.15)

    return ModalityVA(valence, arousal, conf).clamp()


def score_audio(audio_path: Path, *, sr: int = 16000, duration: float = 8.0) -> ModalityVA:
    """librosa 能量 / 谱质心 → (v, a) 代理。"""
    if not audio_path.is_file():
        return ModalityVA(0.0, 0.0, 0.0).clamp()

    try:
        import librosa
    except ImportError:
        return ModalityVA(0.0, 0.0, 0.1).clamp()

    try:
        y, _ = librosa.load(str(audio_path), sr=sr, duration=duration, mono=True)
    except Exception:
        return ModalityVA(0.0, 0.0, 0.0).clamp()

    if y.size == 0:
        return ModalityVA(0.0, 0.0, 0.0).clamp()

    rms = float(np.sqrt(np.mean(y**2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    arousal = _clip(math.tanh(rms * 12 + zcr * 2))
    valence = _clip(math.tanh((cent - 2000) / 2000))
    conf = _clip(0.4 + rms * 2)
    return ModalityVA(valence, arousal, conf).clamp()


def _face_frames(face_path: Path) -> np.ndarray | None:
    if not face_path.is_file():
        return None
    try:
        arr = np.load(face_path, allow_pickle=False)
    except Exception:
        return None
    if arr.ndim != 4:
        return None
    return arr


def score_face_npy(
    face_path: Path,
    *,
    granularity: Literal["fine", "coarse"] = "fine",
) -> ModalityVA:
    """OpenFace 人脸帧序列：fine≈微表情变化，coarse≈宏表情均值。"""
    frames = _face_frames(face_path)
    if frames is None:
        return ModalityVA(0.0, 0.0, 0.0).clamp()

    gray = frames.astype(np.float32).mean(axis=-1)
    valid = gray.sum(axis=(1, 2)) > 1.0
    if not np.any(valid):
        return ModalityVA(0.0, 0.0, 0.0).clamp()

    gray = gray[valid]
    if granularity == "coarse":
        mean_int = float(gray.mean()) / 255.0
        valence = _clip((mean_int - 0.45) * 2.5)
        arousal = _clip(float(gray.std()) / 64.0)
    else:
        if gray.shape[0] < 2:
            diff = 0.0
        else:
            diff = float(np.mean(np.abs(np.diff(gray, axis=0))))
        valence = _clip((float(gray.mean()) / 255.0 - 0.45) * 2.0)
        arousal = _clip(diff / 12.0)

    conf = _clip(0.35 + valid.sum() / max(frames.shape[0], 1) * 0.5)
    return ModalityVA(valence, arousal, conf).clamp()


def score_sample(sample: DatasetSample) -> dict[str, ModalityVA]:
    """对单样本计算四模态 VA 代理。"""
    return {
        "text": score_text(sample.subtitle, openset=sample.openset or None),
        "audio": score_audio(sample.audio_path),
        "face": score_face_npy(sample.face_path, granularity="fine"),
        "frame": score_face_npy(sample.face_path, granularity="coarse"),
    }


def valence_dict(scores: dict[str, ModalityVA]) -> dict[str, float]:
    return {k: v.valence for k, v in scores.items()}
