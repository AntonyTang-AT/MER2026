"""配置加载 smoke test。"""

from src.core.config_loader import get_project_root, load_global_config, load_yaml


def test_project_root_exists():
    assert get_project_root().is_dir()


def test_load_global_config():
    cfg = load_global_config()
    assert "paths" in cfg


def test_load_weight_table():
    wt = load_yaml("routing/weight_table.yaml")
    assert "consistent" in wt
    assert sum(wt["consistent"].values()) == 1.0
