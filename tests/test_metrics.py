"""Unit tests for answer classification and accuracy computation."""

import pytest

from utils.metrics import classify_answer, compute_accuracy, init_results

QUESTION = {
    "database_idx": "0",
    "question_type": "ego_movement",
    "question": "Where is the mirror?",
    "answer_choices": ["A. Left", "B. Right", "C. Behind"],
    "correct_answer": "A. Left",
}


def test_classify_correct():
    assert classify_answer("A. Left", QUESTION) == "correct"


def test_classify_correct_multiline():
    # correct answer must appear on the final line
    assert classify_answer("Hmm, let me think.\nA. Left", QUESTION) == "correct"


def test_classify_wrong():
    assert classify_answer("B. Right", QUESTION) == "wrong"


def test_classify_wrong_multiline():
    assert classify_answer("Maybe.\nB. Right", QUESTION) == "wrong"


def test_classify_out_of_control():
    assert classify_answer("I don't know", QUESTION) == "out of control"


def test_init_results_structure():
    results = init_results(["a", "b"])
    assert results["progress"] == {
        "a": {"correct": [], "wrong": []},
        "b": {"correct": [], "wrong": []},
    }
    assert results["accuracy"]["all"] == 0.0
    assert results["skip_indices"] == []


def test_compute_accuracy():
    progress = {
        "a": {"correct": ["1"], "wrong": ["2"]},
        "b": {"correct": ["3"], "wrong": []},
    }
    acc = compute_accuracy(progress, skip_indices=[], total_questions=3)
    assert acc["all"] == pytest.approx(2 / 3)
