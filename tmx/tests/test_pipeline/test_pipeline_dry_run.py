import json

import numpy as np
import pytest

from src.inference.pipeline import PipelineRunner


@pytest.fixture
def pipeline_config(tmp_path):
    routing_dir = tmp_path / "outputs" / "routing"
    routing_dir.mkdir(parents=True)
    routing_json = routing_dir / "human_routing.json"
    routing_json.write_text("[]", encoding="utf-8")
    return {
        "stages": {
            "routing": {"enabled": True},
            "affectgpt": {"enabled": False, "use_routing_in_prompt": False},
            "openset_extract": {"enabled": False},
            "ensemble": {"enabled": False},
            "submission": {"enabled": True},
            "evaluation": {"enabled": True},
        },
        "artifacts": {
            "routing_dir": str(routing_dir.relative_to(tmp_path)),
            "submission_dir": str((tmp_path / "outputs" / "submissions").relative_to(tmp_path)),
            "timing_dir": str((tmp_path / "outputs" / "pipeline_runs").relative_to(tmp_path)),
        },
        "runtime": {"skip_existing": True, "dry_run": False},
    }


def test_pipeline_dry_run_submit(tmp_path, pipeline_config, monkeypatch):
    monkeypatch.setattr(
        "src.inference.pipeline.get_project_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "src.inference.pipeline.sync_mertools_config",
        lambda verbose=False: None,
    )
    monkeypatch.setattr(
        "src.inference.pipeline.MER2026Index.from_config",
        lambda: _FakeIndex([]),
    )
    monkeypatch.setattr(
        "src.inference.pipeline.run_batch",
        lambda *a, **k: [],
    )

    runner = PipelineRunner(pipeline_config)
    artifacts = runner.run(
        mode="submit",
        dry_run=True,
        skip_stages=["routing"],
        openset_npz=_write_mock_openset(tmp_path),
        profile=True,
    )
    assert artifacts.submission_csv is not None
    assert "total" in artifacts.timing


def test_pipeline_skip_existing_routing(tmp_path, pipeline_config, monkeypatch):
    monkeypatch.setattr("src.inference.pipeline.get_project_root", lambda: tmp_path)
    monkeypatch.setattr("src.inference.pipeline.sync_mertools_config", lambda verbose=False: None)

    routing_path = tmp_path / pipeline_config["artifacts"]["routing_dir"] / "human_routing.json"
    routing_path.write_text("[{\"name\":\"x\"}]", encoding="utf-8")

    called = {"run_batch": False}

    def _fake_batch(*args, **kwargs):
        called["run_batch"] = True
        return []

    monkeypatch.setattr("src.inference.pipeline.run_batch", _fake_batch)

    runner = PipelineRunner({**pipeline_config, "stages": {**pipeline_config["stages"], "submission": {"enabled": False}, "evaluation": {"enabled": False}}})
    artifacts = runner.run(mode="eval", skip_existing=True, dry_run=True)
    assert artifacts.routing_json == routing_path
    assert called["run_batch"] is False


class _FakeIndex:
    def __init__(self, samples):
        self.samples = samples

    def load_split(self, split, limit=None, check_media=False):
        return self.samples


def _write_mock_openset(tmp_path):
    path = tmp_path / "mock-openset.npz"
    np.savez_compressed(path, filenames=["n1"], fileitems=["[happy]"])
    return path
