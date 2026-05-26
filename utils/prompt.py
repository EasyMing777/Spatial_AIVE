"""Prompt formatting utilities for VLM-based spatial reasoning.

This module constructs (system_prompt, content) tuples consumed by ModelAdapter
instances. Content is a list of (text, image_path) pairs describing the
conversation turn.
"""

from typing import Any, Dict, List, Optional, Tuple


def _build_image_content(images: List[str], initial_label: str = "Image 1") -> List[Tuple[str, Optional[str]]]:
    """Build the image portion of a content list.

    Args:
        images: List of image file paths.
        initial_label: Label for the first image.

    Returns:
        List of (text, image_path) tuples.
    """
    content: List[Tuple[str, Optional[str]]] = []
    if not images:
        content.append(("No image provided.\n\n", None))
        return content

    for idx, img_path in enumerate(images):
        content.append((f"Image {idx + 1}:", img_path))
    content.append((f"\n{initial_label} is the current view.\n", None))
    if len(images) > 1:
        content.append(("Image 2 is an additional reference view.\n", None))
    return content


def _build_answer_choices_block(answer_choices: List[str]) -> str:
    """Format answer choices into a text block."""
    lines = ["Answer Choices:"]
    lines.extend(f"  - {c}" for c in answer_choices)
    lines.append("")
    return "\n".join(lines)


def format_answer_question(
    question: str,
    answer_choices: List[str],
    images: List[str],
    sys_prompt: str,
) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
    """Format a direct VQA prompt (no exploration context)."""
    content = list(_build_image_content(images))
    content.append((f"Question: {question}\n\n", None))
    content.append((_build_answer_choices_block(answer_choices), None))
    content.append(("Output the exact answer from the choices.\nAnswer: ", None))
    return sys_prompt, content


def format_action_planing(
    question: str,
    answer_choices: List[str],
    images: List[str],
    step_idx: int,
    exploration_history: List[Dict[str, Any]],
    sys_prompt: str,
) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
    """Format an action-planning prompt with exploration history."""
    content = list(_build_image_content(images))
    content.append((f"Question: {question}\n\n", None))

    if exploration_history:
        content.append(("Exploration history (actions already taken):\n", None))
        for hist in exploration_history:
            if hist["step"] < step_idx:
                content.append((f"  Step {hist['step']}: {', '.join(hist['action_sequence'])}\n", None))
        content.append(("You have seen the results of these actions in the previous images.\n", None))

    return sys_prompt, content


def format_discriminator(
    question: str,
    answer_choices: List[str],
    images: List[str],
    exploration_history: Optional[List[Dict[str, Any]]],
    sys_prompt: str,
    step_idx: Optional[int] = None,
) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
    """Format a discriminator prompt (explore/stop gate)."""
    content = list(_build_image_content(images))

    if exploration_history:
        last_hist = exploration_history[-1]
        action_str = ", ".join(last_hist["action_sequence"])
        content.append((f"Latest action: {action_str}", last_hist["result_image_path"]))
        if len(exploration_history) > 1:
            prior_actions = "; ".join(
                ", ".join(h["action_sequence"]) for h in exploration_history[:-1]
            )
            content.append((f"Prior actions (no images): {prior_actions}\n", None))
        content.append(("The latest generated view from the most recent action is shown above.\n", None))

    content.append((f"Question: {question}\n", None))
    return sys_prompt, content


def format_answer_question_exploration(
    question: str,
    answer_choices: List[str],
    images: List[str],
    exploration_history: List[Dict[str, Any]],
    sys_prompt: str,
) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
    """Format a VQA prompt that includes the full exploration trajectory."""
    content: List[Tuple[str, Optional[str]]] = []
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
        content.append((
            "The historical action trajectories and corresponding camera perspectives are shown above.\n",
            None,
        ))

    content.append((f"Question: {question}\n\n", None))
    content.append((_build_answer_choices_block(answer_choices), None))
    content.append(("Output the exact answer from the choices.\nAnswer: ", None))

    return sys_prompt, content
