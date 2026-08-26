"""Unit tests for dataset-format normalisation (SAT + SpaThor-1K)."""

import pytest

from pipelines.AIVE_baseline import _normalise_question


def test_sat_format():
    raw = {
        "database_idx": 42,
        "question_type": "ego_movement",
        "question": "Where is the lamp?",
        "answer_choices": ["A. Left", "B. Right"],
        "correct_answer": "A. Left",
        "img_paths": ["img_0.png"],
    }
    out = _normalise_question(raw)
    assert out["database_idx"] == "42"
    assert out["question_type"] == "ego_movement"
    assert out["answer_choices"] == ["A. Left", "B. Right"]
    assert out["correct_answer"] == "A. Left"
    assert out["img_paths"] == ["img_0.png"]


def test_spathor1k_format():
    raw = {
        "id": 7,
        "type": "occlusion",
        "question": "What is hidden behind the TV stand?",
        "choices": ["A. Trash can", "B. Carton"],
        "answer": "A. Trash can",
        "image": "frame_0.png",
    }
    out = _normalise_question(raw)
    assert out["database_idx"] == "7"
    assert out["question_type"] == "occlusion"
    assert out["correct_answer"] == "A. Trash can"
    assert out["img_paths"] == ["frame_0.png"]


def test_single_image_coerced_to_list():
    raw = {
        "id": 1,
        "type": "distance",
        "question": "Q?",
        "choices": ["A", "B"],
        "answer": "A",
        "image_path": "frame.png",
    }
    out = _normalise_question(raw)
    assert out["img_paths"] == ["frame.png"]


def test_missing_id_raises():
    with pytest.raises(ValueError):
        _normalise_question({"question": "no id here"})


def test_missing_type_defaults_to_other():
    out = _normalise_question({"id": 1, "question": "Q?", "choices": ["A"], "answer": "A"})
    assert out["question_type"] == "other"
