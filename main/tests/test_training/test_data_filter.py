"""data_filter 单元测试。"""

from pathlib import Path

import pandas as pd

from src.training.data_filter import (
    FilterConfig,
    MediaRoots,
    apply_filters,
    parse_openset,
    run_filter,
)


def test_parse_openset_bracket_string():
    assert parse_openset("[concern, pessimism]") == ["concern", "pessimism"]


def test_parse_openset_empty():
    assert parse_openset("") == []
    assert parse_openset("[]") == []


def test_apply_filters_drops_empty_and_human_overlap(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"name": "a", "openset": "[happy]"},
            {"name": "b", "openset": "[]"},
            {"name": "c", "openset": "[sad, angry, worried, tense, upset, fear, grief, pain, shock, rage, dread, sorrow, alarm, panic, dread2]"},
        ],
        columns=["name", "openset"],
    )
    cfg = FilterConfig(max_labels=14, exclude_human_names=True)
    out, summary = apply_filters(df, cfg, human_names={"a"})
    assert summary.output_count == 0
    assert summary.removed["empty_openset"] == 1
    assert summary.removed["overlap_human"] == 1
    assert summary.removed["too_many_labels"] == 1


def test_apply_filters_require_media(tmp_path: Path):
    audio = tmp_path / "audio"
    video = tmp_path / "video"
    face = tmp_path / "openface_face"
    audio.mkdir()
    video.mkdir()
    face.mkdir()

    name = "sample_001"
    (audio / f"{name}.wav").write_text("x", encoding="utf-8")
    (video / f"{name}.mp4").write_text("x", encoding="utf-8")
    (face / name).mkdir()
    (face / name / f"{name}.npy").write_text("x", encoding="utf-8")

    df = pd.DataFrame([{"name": name, "openset": "[happy]"}], columns=["name", "openset"])
    roots = MediaRoots(audio=audio, video=video, face=face)
    out, summary = apply_filters(
        df,
        FilterConfig(require_media=True),
        media_roots=roots,
    )
    assert len(out) == 1
    assert summary.output_count == 1


def test_run_filter_on_real_csv_if_present():
    from src.training.data_filter import default_mercaptionplus_csv, default_data_root

    csv_path = default_mercaptionplus_csv(default_data_root())
    if not csv_path.is_file():
        return

    out_path, report = run_filter(
        input_csv=csv_path,
        output_csv=csv_path.parent / "track2_train_mercaptionplus_filtered_test.csv",
        report_json=csv_path.parent / "mercaptionplus_filter_report_test.json",
        cfg=FilterConfig(require_media=False),
    )
    assert out_path.is_file()
    assert report["output_count"] < report["input_count"]
    assert report["removed"]["empty_openset"] >= 500
    assert report["removed"]["overlap_human"] >= 500
