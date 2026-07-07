"""epoch_sweep 规划逻辑测试。"""

from pathlib import Path

from src.training.epoch_sweep import (
    assign_epochs_to_gpus,
    list_checkpoints,
    parse_epoch_from_checkpoint,
    plan_epochs,
    select_ckpt_root,
    write_sweep_report,
    EpochSweepRow,
)


def test_parse_epoch_from_checkpoint():
    path = Path("checkpoint_000043_loss_0.012.pth")
    assert parse_epoch_from_checkpoint(path) == 43


def test_plan_epochs_respects_skip_and_available():
    epochs = plan_epochs(start=10, end=60, skip=5, available={10, 15, 20, 55, 57})
    assert epochs == [10, 15, 20, 55]


def test_assign_epochs_to_gpus_round_robin():
    epochs = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    gpus = ["0", "1", "2", "3", "4"]
    assignment = assign_epochs_to_gpus(epochs, gpus)
    assert assignment["0"] == [10, 35, 60]
    assert assignment["1"] == [15, 40]
    assert sum(len(v) for v in assignment.values()) == len(epochs)


def test_select_ckpt_root_prefers_most_checkpoints(tmp_path: Path, monkeypatch):
    cfg_name = "human_outputhybird_bestsetup_bestfusion_face_lz"
    root_a = tmp_path / "output" / cfg_name / f"{cfg_name}_a"
    root_b = tmp_path / "output" / cfg_name / f"{cfg_name}_b"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)
    (root_a / "checkpoint_000010_loss_0.100.pth").write_text("a", encoding="utf-8")
    (root_b / "checkpoint_000010_loss_0.100.pth").write_text("b", encoding="utf-8")
    (root_b / "checkpoint_000020_loss_0.050.pth").write_text("b", encoding="utf-8")

    monkeypatch.setattr(
        "src.training.epoch_sweep.find_train_cfg_name",
        lambda _run: cfg_name,
    )
    selected = select_ckpt_root(tmp_path, "human")
    assert selected.name.endswith("_b")


def test_list_checkpoints(tmp_path: Path):
    ckpt_root = tmp_path / "ckpts"
    ckpt_root.mkdir()
    (ckpt_root / "checkpoint_000010_loss_0.100.pth").write_text("x", encoding="utf-8")
    (ckpt_root / "checkpoint_000020_loss_0.050.pth").write_text("x", encoding="utf-8")
    mapping = list_checkpoints(ckpt_root)
    assert mapping[10].name.startswith("checkpoint_000010")
    assert mapping[20].name.startswith("checkpoint_000020")


def test_write_sweep_report(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.training.epoch_sweep.get_project_root", lambda: tmp_path)
    rows = [
        EpochSweepRow(epoch=10, checkpoint="ckpt10", ew_f1=0.55, status="evaluated"),
        EpochSweepRow(epoch=20, checkpoint="ckpt20", ew_f1=0.58, status="evaluated"),
    ]
    md_path, json_path = write_sweep_report(rows, train_run="human")
    assert md_path.is_file()
    assert json_path.is_file()
    assert (tmp_path / "experiments/exp001_baseline/best_epoch.json").is_file()
