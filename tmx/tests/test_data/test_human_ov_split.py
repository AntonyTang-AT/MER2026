"""human_ov_split tests."""

from src.data.human_ov_split import load_human_names, make_split


def test_split_sizes_and_reproducibility():
    names = load_human_names()
    assert len(names) == 1532

    train1, val1 = make_split()
    train2, val2 = make_split()
    assert len(val1) == 306
    assert len(train1) == 1226
    assert len(train1) + len(val1) == 1532
    assert val1 == val2
    assert train1 == train2
