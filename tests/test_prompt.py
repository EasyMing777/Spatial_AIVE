"""Unit tests for prompt formatting helpers."""

from utils.prompt import (
    format_action_planing,
    format_answer_question,
    format_answer_question_exploration,
    format_discriminator,
)

SYS = "You are an AI assistant for spatial reasoning."
Q = "Where is the mirror?"
CHOICES = ["A. Left", "B. Right"]
IMAGES = ["/img/a.png", "/img/b.png"]


def _content_text(content):
    return " ".join(t for t, _ in content if t)


def test_answer_question_contains_image_and_question():
    sys_prompt, content = format_answer_question(
        question=Q, answer_choices=CHOICES, images=IMAGES, sys_prompt=SYS
    )
    assert sys_prompt == SYS
    text = _content_text(content)
    assert "Image 1:" in text and "Image 2:" in text
    assert Q in text and "A. Left" in text
    # image paths are wired as the second element of each tuple
    assert any(img == "/img/a.png" for _, img in content)


def test_action_planing_includes_history():
    history = [{"step": 0, "action_sequence": ["turn-left 9"], "result_image_path": "/img/x.png"}]
    _, content = format_action_planing(
        question=Q,
        answer_choices=CHOICES,
        images=IMAGES,
        step_idx=1,
        exploration_history=history,
        sys_prompt=SYS,
    )
    text = _content_text(content)
    assert "turn-left 9" in text


def test_discriminator_labels_initial_view():
    history = [
        {
            "step": 0,
            "action_sequence": ["move-forward 0.5"],
            "result_image_path": "/img/frame1.png",
        }
    ]
    _, content = format_discriminator(
        question=Q,
        answer_choices=CHOICES,
        images=["/img/i0.png"],
        exploration_history=history,
        sys_prompt=SYS,
        step_idx=0,
    )
    text = _content_text(content)
    assert "INITIAL view" in text
    assert any(img == "/img/i0.png" for _, img in content)


def test_discriminator_attaches_latest_view():
    history = [
        {
            "step": 0,
            "action_sequence": ["turn-left 9", "move-forward 0.5"],
            "result_image_path": "/img/view.png",
        }
    ]
    _, content = format_discriminator(
        question=Q,
        answer_choices=CHOICES,
        images=IMAGES,
        exploration_history=history,
        sys_prompt=SYS,
        step_idx=0,
    )
    # the latest generated view should be paired with the action text
    assert any(img == "/img/view.png" for _, img in content)


def test_answer_question_exploration_lists_trajectory():
    history = [
        {
            "step": 0,
            "action_sequence": ["move-forward 0.5"],
            "result_image_path": "/img/frame1.png",
        }
    ]
    _, content = format_answer_question_exploration(
        question=Q,
        answer_choices=CHOICES,
        images=IMAGES,
        exploration_history=history,
        sys_prompt=SYS,
    )
    text = _content_text(content)
    assert "move-forward 0.5" in text and Q in text
    assert any(img == "/img/frame1.png" for _, img in content)


def test_no_images_handled():
    _, content = format_answer_question(
        question=Q, answer_choices=CHOICES, images=[], sys_prompt=SYS
    )
    assert "No image provided" in _content_text(content)
