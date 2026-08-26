"""Prompt formatting utilities for VLM-based spatial reasoning.

This module constructs (system_prompt, content) tuples consumed by ModelAdapter
instances. Content is a list of (text, image_path) pairs describing the
conversation turn.
"""

from typing import Any


def _build_image_content(
    images: list[str], initial_label: str = "Image 1"
) -> list[tuple[str, str | None]]:
    """Build the image portion of a content list.

    Args:
        images: List of image file paths.
        initial_label: Label for the first image.

    Returns:
        List of (text, image_path) tuples.
    """
    content: list[tuple[str, str | None]] = []
    if not images:
        content.append(("No image provided.\n\n", None))
        return content

    for idx, img_path in enumerate(images):
        content.append((f"Image {idx + 1}:", img_path))
    content.append((f"\n{initial_label} is the current view.\n", None))
    if len(images) > 1:
        content.append(("Image 2 is an additional reference view.\n", None))
    return content


def _build_answer_choices_block(answer_choices: list[str]) -> str:
    """Format answer choices into a text block."""
    lines = ["Answer Choices:"]
    lines.extend(f"  - {c}" for c in answer_choices)
    lines.append("")
    return "\n".join(lines)


def format_answer_question(
    question: str,
    answer_choices: list[str],
    images: list[str],
    sys_prompt: str,
) -> tuple[str, list[tuple[str, str | None]]]:
    """Format a direct VQA prompt (no exploration context)."""
    content = list(_build_image_content(images))
    content.append((f"Question: {question}\n\n", None))
    content.append((_build_answer_choices_block(answer_choices), None))
    content.append(("Output the exact answer from the choices.\nAnswer: ", None))
    return sys_prompt, content


def format_action_planing(
    question: str,
    answer_choices: list[str],
    images: list[str],
    step_idx: int,
    exploration_history: list[dict[str, Any]],
    sys_prompt: str,
) -> tuple[str, list[tuple[str, str | None]]]:
    """Format an action-planning prompt with exploration history."""
    content = list(_build_image_content(images))
    content.append((f"Question: {question}\n\n", None))

    if exploration_history:
        content.append(("Exploration history (actions already taken):\n", None))
        for hist in exploration_history:
            if hist["step"] < step_idx:
                content.append(
                    (f"  Step {hist['step']}: {', '.join(hist['action_sequence'])}\n", None)
                )
        content.append(
            ("You have seen the results of these actions in the previous images.\n", None)
        )

    return sys_prompt, content


def format_discriminator(
    question: str,
    answer_choices: list[str],
    images: list[str],
    exploration_history: list[dict[str, Any]] | None,
    sys_prompt: str,
    step_idx: int | None = None,
) -> tuple[str, list[tuple[str, str | None]]]:
    """Format a discriminator prompt (explore/stop gate).

    Per Algorithm 1 (line 11) the Checker evaluates the *initial* observation
    ``i0`` together with the accumulated exploration history ``H``:
    ``z = V_check(q, i0, H)``. ``images`` therefore carries the initial view
    (plus an optional reference view), while the latest imagined view is
    injected from the exploration history.
    """
    content: list[tuple[str, str | None]] = []
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is the INITIAL view — the original observation.\n", None))
        if len(images) > 1:
            content.append(("Image 2 is an additional reference view.\n", None))
    else:
        content.append(("No image provided.\n\n", None))

    if exploration_history:
        last_hist = exploration_history[-1]
        action_str = ", ".join(last_hist["action_sequence"])
        content.append((f"Latest action: {action_str}", last_hist["result_image_path"]))
        if len(exploration_history) > 1:
            prior_actions = "; ".join(
                ", ".join(h["action_sequence"]) for h in exploration_history[:-1]
            )
            content.append((f"Prior actions (no images): {prior_actions}\n", None))
        content.append(
            ("The latest generated view after the most recent action is shown above.\n", None)
        )

    content.append((f"Question: {question}\n", None))
    return sys_prompt, content


def format_answer_question_exploration(
    question: str,
    answer_choices: list[str],
    images: list[str],
    exploration_history: list[dict[str, Any]],
    sys_prompt: str,
) -> tuple[str, list[tuple[str, str | None]]]:
    """Format a VQA prompt that includes the full exploration trajectory."""
    content: list[tuple[str, str | None]] = []
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is the INITIAL VIEW — the original observation.\n", None))
        if len(images) > 1:
            content.append(("Image 2 is an additional reference view.\n", None))
    else:
        content.append(("No image provided.\n\n", None))

    if exploration_history:
        for idx, hist in enumerate(exploration_history):
            action_str = ", ".join(hist["action_sequence"])
            content.append((f"Action {idx + 1}: {action_str}", hist["result_image_path"]))
        content.append(
            (
                "The historical action trajectories and corresponding camera perspectives are shown above.\n",
                None,
            )
        )

    content.append((f"Question: {question}\n\n", None))
    content.append((_build_answer_choices_block(answer_choices), None))
    content.append(("Output the exact answer from the choices.\nAnswer: ", None))

    return sys_prompt, content
