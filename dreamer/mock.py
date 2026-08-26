"""Deterministic placeholder Dreamer for development and end-to-end testing.

The MockDreamer lets the full AIVE loop (Checker -> Planner -> *Dreamer* ->
Re-check -> Answerer) run without a trained generative world model. It does not
learn anything; it simply returns a view for the next step:

* ``identity`` — returns the input view unchanged (a cheap copy);
* ``shift`` — applies a small affine translation to the input, roughly
  mimicking the parallax of a forward camera motion.

Replace this class with :class:`dreamer.wan2_2.Wan2_2Dreamer` (or your own
``BaseDreamer`` subclass) once a checkpoint is available.
"""

from __future__ import annotations

import os

import cv2

from dreamer.base import BaseDreamer


class MockDreamer(BaseDreamer):
    """Placeholder world model with deterministic, dependency-free behaviour."""

    name = "mock"

    def __init__(self, mode: str = "identity", **kwargs: object) -> None:
        super().__init__(**kwargs)
        if mode not in ("identity", "shift"):
            raise ValueError(f"Unknown mock mode {mode!r}; expected 'identity' or 'shift'.")
        self.mode = mode

    def generate(
        self,
        image_path: str,
        action_sequence: list[str],
        save_dir: str,
        step_idx: int = 0,
    ) -> str | None:
        """Return a deterministic 'imagined' next view.

        Parameters
        ----------
        image_path : str
            Path to the current observation.
        action_sequence : list of str
            Planned actions (informational only for the mock).
        save_dir : str
            Output directory for the generated frame.
        step_idx : int
            Exploration step index.

        Returns
        -------
        str or None
            Path to the saved next view, or ``None`` if the input cannot be read.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        if self.mode == "shift":
            # Simulate a subtle forward parallax with a small crop-and-resize.
            h, w = img.shape[:2]
            top, left = int(0.02 * h), int(0.02 * w)
            img = img[top:, left:]

        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"imagined_step_{step_idx}.png")
        cv2.imwrite(out_path, img)
        return out_path
