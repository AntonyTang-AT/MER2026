import numpy as np

from src.inference.affectgpt_runner import find_latest_openset_npz, find_latest_reason_npz


def test_find_latest_reason_npz_filters_openset(tmp_path, monkeypatch):
    root = tmp_path / "mertools"
    out = root / "output" / "results-mer2026ov-mer2026ov"
    out.mkdir(parents=True)
    reason = out / "checkpoint_000010.npz"
    openset = out / "checkpoint_000010-openset.npz"
    np.savez_compressed(reason, name2reason={"a": "x"})
    np.savez_compressed(openset, filenames=["a"], fileitems=["[]"])

    monkeypatch.setattr(
        "src.inference.affectgpt_runner.get_paths",
        lambda: {"mertools_root": root, "project_root": tmp_path},
    )
    monkeypatch.setattr(
        "src.inference.affectgpt_runner.load_yaml",
        lambda name: {"inference": {"base_root": "output/results-mer2026ov"}} if name == "baseline.yaml" else {},
    )

    latest = find_latest_reason_npz(mertools_root=root, prompts=False)
    assert latest == reason


def test_find_latest_openset_from_reason_sibling(tmp_path, monkeypatch):
    root = tmp_path / "mertools"
    out = root / "output" / "results-mer2026ov-mer2026ov"
    out.mkdir(parents=True)
    reason = out / "checkpoint_000010.npz"
    openset = out / "checkpoint_000010-openset.npz"
    np.savez_compressed(reason, name2reason={"a": "x"})
    np.savez_compressed(openset, filenames=["a"], fileitems=["[happy]"])

    monkeypatch.setattr(
        "src.inference.affectgpt_runner.get_paths",
        lambda: {"mertools_root": root, "project_root": tmp_path},
    )
    monkeypatch.setattr(
        "src.inference.affectgpt_runner.load_yaml",
        lambda name: {"inference": {"base_root": "output/results-mer2026ov"}} if name == "baseline.yaml" else {},
    )

    assert find_latest_openset_npz(reason) == openset
