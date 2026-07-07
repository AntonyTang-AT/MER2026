"""MERTools path sync tests."""

from src.training.mertools_paths import get_paths, sync_mertools_config


def test_sync_mertools_config():
    result = sync_mertools_config(verbose=False)
    assert "mertools_root" in result
    assert "data_root" in result

    paths = get_paths()
    config_py = paths["mertools_root"] / "config.py"
    text = config_py.read_text(encoding="utf-8")
    assert "xxx/dataset" not in text
    assert str(paths["data_root"]).replace("\\", "/") in text

    models_link = paths["mertools_root"] / "models"
    assert models_link.exists()
