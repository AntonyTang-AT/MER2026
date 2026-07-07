"""filter_train_annotations tests."""

from src.data.human_ov_split import (
    filter_train_annotations,
    load_human_names,
    load_val_names,
    val_name_set,
)


def test_filter_train_annotations_excludes_val():
    val_names = load_val_names()
    assert len(val_names) == 306
    annotation = [{"name": name, "subtitle": "", "ovlabel": "happy"} for name in load_human_names()]
    filtered = filter_train_annotations(annotation)
    assert len(filtered) == 1226
    assert not val_name_set() & {item["name"] for item in filtered}
