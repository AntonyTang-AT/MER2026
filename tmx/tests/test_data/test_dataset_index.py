"""dataset_index smoke tests."""

from src.data.dataset_index import MER2026Index


def test_iter_human_first_10():
    idx = MER2026Index.from_config()
    samples = idx.iter_human(10)
    assert len(samples) == 10
    for sample in samples:
        assert sample.name
        assert isinstance(sample.openset, list)
        assert len(sample.openset) >= 1
        assert sample.video_path.name.endswith(".mp4")
        assert sample.audio_path.name.endswith(".wav")
        assert sample.face_path.name.endswith(".npy")
