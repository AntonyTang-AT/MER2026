"""MER2026 CSV 与媒体路径索引。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.core.config_loader import get_project_root, load_global_config, load_yaml
SplitName = Literal["human", "mercaptionplus", "candidate"]


def _parse_openset(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    return [
        item.strip().lower()
        for item in re.split(r"['\",]", text)
        if item.strip() not in ("", ",")
    ]


@dataclass
class DatasetSample:
    name: str
    openset: list[str]
    subtitle: str
    video_path: Path
    audio_path: Path
    face_path: Path

    @property
    def media_exists(self) -> bool:
        return self.video_path.is_file() and self.audio_path.is_file() and self.face_path.is_file()


class MER2026Index:
    """MER2026 数据集索引：CSV 标签 + 媒体路径 + 字幕。"""

    _SPLIT_FILES: dict[SplitName, str] = {
        "human": "train_human",
        "mercaptionplus": "train_mercaptionplus",
        "candidate": "candidate",
    }

    def __init__(self, data_root: Path, dataset_cfg: dict) -> None:
        self.data_root = data_root.resolve()
        self.dataset_cfg = dataset_cfg
        self._subtitle_map: dict[str, str] | None = None

    @classmethod
    def from_config(cls) -> MER2026Index:
        global_cfg = load_global_config()
        dataset_cfg = load_yaml("dataset.yaml")
        data_root = (get_project_root() / global_cfg["paths"]["data_root"]).resolve()
        return cls(data_root=data_root, dataset_cfg=dataset_cfg)

    def _csv_path(self, split: SplitName) -> Path:
        file_key = self._SPLIT_FILES[split]
        filename = self.dataset_cfg["files"][file_key]
        path = self.data_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found for split '{split}': {path}")
        return path

    def _subtitle_csv(self) -> Path:
        filename = self.dataset_cfg["files"]["subtitle"]
        return self.data_root / filename

    def _load_subtitles(self) -> dict[str, str]:
        if self._subtitle_map is not None:
            return self._subtitle_map

        subtitle_col = self.dataset_cfg["csv_columns"]["subtitle_en"]
        path = self._subtitle_csv()
        if not path.is_file():
            self._subtitle_map = {}
            return self._subtitle_map

        df = pd.read_csv(path)
        name2subtitle: dict[str, str] = {}
        for _, row in df.iterrows():
            name = str(row["name"])
            subtitle = row.get(subtitle_col, "")
            if pd.isna(subtitle):
                subtitle = ""
            name2subtitle[name] = str(subtitle)
        self._subtitle_map = name2subtitle
        return self._subtitle_map

    def resolve_paths(self, name: str) -> tuple[Path, Path, Path]:
        dirs = self.dataset_cfg["dirs"]
        video = self.data_root / dirs["video"] / f"{name}.mp4"
        audio = self.data_root / dirs["audio"] / f"{name}.wav"
        face = self.data_root / dirs["openface_face"] / name / f"{name}.npy"
        return video, audio, face

    def load_split(
        self,
        split: SplitName,
        *,
        check_media: bool = False,
        limit: int | None = None,
    ) -> list[DatasetSample]:
        csv_path = self._csv_path(split)
        df = pd.read_csv(csv_path)
        name_col = self.dataset_cfg["csv_columns"]["name"]
        openset_col = self.dataset_cfg["csv_columns"]["openset"]
        subtitles = self._load_subtitles()

        samples: list[DatasetSample] = []
        for _, row in df.iterrows():
            name = str(row[name_col])
            if split == "candidate":
                openset: list[str] = []
            else:
                openset = _parse_openset(row.get(openset_col, ""))

            video, audio, face = self.resolve_paths(name)
            sample = DatasetSample(
                name=name,
                openset=openset,
                subtitle=subtitles.get(name, ""),
                video_path=video,
                audio_path=audio,
                face_path=face,
            )
            if check_media and not sample.media_exists:
                raise FileNotFoundError(
                    f"Missing media for {name}: "
                    f"video={sample.video_path.exists()} "
                    f"audio={sample.audio_path.exists()} "
                    f"face={sample.face_path.exists()}"
                )
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
        return samples

    def iter_human(self, n: int = 10, **kwargs: object) -> list[DatasetSample]:
        return self.load_split("human", limit=n, **kwargs)  # type: ignore[arg-type]
