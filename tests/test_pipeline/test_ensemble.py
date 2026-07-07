from src.inference.ensemble import merge_openset_npz, merge_openset_predictions


def test_label_union_merges_and_deduplicates():
    preds = [
        {"s1": "[happy, sad]"},
        {"s1": "[happy, angry]"},
    ]
    merged = merge_openset_predictions(preds, strategy="label_union")
    assert merged["s1"] == "[happy, sad, angry]"


def test_majority_vote_requires_min_votes():
    preds = [
        {"s1": "[happy, sad]"},
        {"s1": "[happy, angry]"},
    ]
    merged = merge_openset_predictions(preds, strategy="majority_vote", min_votes=2)
    assert merged["s1"] == "[happy]"


def test_merge_openset_npz_from_files(tmp_path):
    import numpy as np

    p1 = tmp_path / "a-openset.npz"
    p2 = tmp_path / "b-openset.npz"
    np.savez_compressed(p1, filenames=["n1"], fileitems=["[happy]"])
    np.savez_compressed(p2, filenames=["n1"], fileitems=["[sad]"])
    merged = merge_openset_npz([p1, p2], strategy="label_union")
    assert merged["n1"] == "[happy, sad]"
