import numpy as np
import pandas as pd
import pytest

from src.inference.pipeline import PipelineRunner


def test_pipeline_submission_shape(tmp_path, monkeypatch):
    data_root = tmp_path / "data" / "mer2026-dataset"
    data_root.mkdir(parents=True)
    candidate = data_root / "track1_track2_candidate.csv"
    df = pd.DataFrame({"name": [f"s{i}" for i in range(20000)], "openset": [""] * 20000})
    df.to_csv(candidate, index=False)

    openset = tmp_path / "pred-openset.npz"
    np.savez_compressed(
        openset,
        filenames=["s0", "s1"],
        fileitems=["[happy]", "[sad]"],
    )

    sub_dir = tmp_path / "outputs" / "submissions"
    config = {
        "stages": {
            "routing": {"enabled": False},
            "affectgpt": {"enabled": False, "use_routing_in_prompt": False},
            "openset_extract": {"enabled": False},
            "ensemble": {"enabled": False},
            "submission": {"enabled": True},
            "evaluation": {"enabled": False},
        },
        "artifacts": {
            "routing_dir": "outputs/routing",
            "submission_dir": "outputs/submissions",
            "timing_dir": "outputs/pipeline_runs",
        },
        "runtime": {"skip_existing": False, "dry_run": False},
    }

    monkeypatch.setattr("src.inference.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("src.inference.pipeline.sync_mertools_config", lambda verbose=False: None)
    monkeypatch.setattr(
        "src.data.submission_formatter.get_project_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "src.data.submission_formatter.load_global_config",
        lambda: {"paths": {"data_root": "data/mer2026-dataset", "outputs_root": "outputs"}},
    )

    runner = PipelineRunner(config)
    out = tmp_path / "outputs" / "submissions" / "track2_test.csv"
    artifacts = runner.run(
        mode="submit",
        openset_npz=openset,
        submission_out=out,
        skip_stages=["routing", "affectgpt", "openset"],
    )
    assert artifacts.submission_csv == out
    result = pd.read_csv(out)
    assert len(result) == 20000
    assert list(result.columns) == ["name", "openset"]
    assert result.loc[result["name"] == "s0", "openset"].iloc[0] == "[happy]"
