"""submission_formatter tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.submission_formatter import format_submission
from src.core.config_loader import get_project_root


def test_submission_csv_shape():
    root = get_project_root()
    candidate = root / "data/mer2026-dataset/track1_track2_candidate.csv"
    df = pd.read_csv(candidate)
    names = df["name"].astype(str).tolist()[:5]

    name2pred = {name: "happy,joyful" for name in names}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sub.csv"
        format_submission(
            name2pred,
            candidate_csv=candidate,
            out_path=out,
        )
        result = pd.read_csv(out)
        assert list(result.columns) == ["name", "openset"]
        assert len(result) == 20000
        assert result.iloc[0]["name"] == names[0]
