from src.routing.expert_rules import classify_from_valences
from src.routing.modality_scorer import ModalityVA, score_text
from src.routing.va_distance import compute_distances
from src.routing.weight_selector import normalize_weights, select_weights


def test_score_text_positive():
    va = score_text("I am so happy and wonderful today!")
    assert va.valence > 0.2
    assert 0.0 <= va.confidence <= 1.0


def test_score_text_empty():
    va = score_text("")
    assert va.valence == 0.0
    assert va.confidence <= 0.3


def test_classify_masking():
    r = classify_from_valences(text_v=0.8, audio_v=0.0, frame_v=0.0, face_v=-0.5)
    assert r.contradiction_type == "masking"


def test_classify_sarcasm():
    r = classify_from_valences(text_v=0.5, audio_v=-0.5, frame_v=0.0, face_v=0.0)
    assert r.contradiction_type == "sarcasm"


def test_classify_consistent():
    r = classify_from_valences(text_v=0.1, audio_v=0.1, frame_v=0.1, face_v=0.1)
    assert r.contradiction_type == "consistent"


def test_distance_matrix_symmetric():
    scores = {
        "text": ModalityVA(0.8, 0.2, 0.9),
        "audio": ModalityVA(-0.5, 0.6, 0.8),
        "face": ModalityVA(-0.4, 0.5, 0.7),
        "frame": ModalityVA(0.0, 0.1, 0.6),
    }
    d = compute_distances(scores)
    assert d.matrix.shape == (4, 4)
    assert d.max_distance >= 0.0
    assert d.matrix[0, 1] == d.matrix[1, 0]


def test_weights_sum_to_one():
    w, conf = select_weights("masking", intensity=0.8)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert 0.0 < conf <= 1.0


def test_normalize_weights():
    w = normalize_weights({"text": 1, "audio": 1, "face": 1, "frame": 1})
    assert all(abs(v - 0.25) < 1e-6 for v in w.values())
