"""Action-conditioned Dreamer built on Wan2.2-TI2V (extension point).

This class is the intended deployment target for the trained world model
described in Section 4.3 of the paper:

    * backbone  — Wan2.2-TI2V-5B, weights kept frozen;
    * adapters  — a small trainable camera encoder + attention projector
                  injected into each block (~284M params, ~5% of the network);
    * condition — the relative camera pose :math:`\\Delta P_t = P_{t-1}^{-1} P_t
                  \\in SE(3)` derived from the discrete action trajectory.

Training the Dreamer requires the *Trajectory Records* of the SpaThor dataset
and is out of scope for this inference-only repository. Once a checkpoint is
available, set the ``AIVE_WAN_CKPT_PATH`` environment variable, implement
:meth:`Wan2_2Dreamer.generate` below, and swap it in via ``--dreamer_type wan2_2``.
"""

from __future__ import annotations

import os

from dreamer.base import BaseDreamer, actions_to_relative_pose


class Wan2_2Dreamer(BaseDreamer):
    """Wan2.2-TI2V-based Dreamer (not yet implemented).

    Parameters
    ----------
    ckpt_path : str, optional
        Path to a trained Dreamer checkpoint. Falls back to the
        ``AIVE_WAN_CKPT_PATH`` environment variable.
    **kwargs
        Additional configuration forwarded to the underlying generator.
    """

    name = "wan2_2"

    def __init__(self, ckpt_path: str | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ckpt_path = ckpt_path or os.getenv("AIVE_WAN_CKPT_PATH")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Wan2.2-TI2V backbone and inject camera-conditioning layers.

        TODO(dreamer): implement.
            * ``AutoModelForImageTextToText``/Wan2.2 loader from ``self.ckpt_path``.
            * inject a ``Linear(12 -> dim)`` camera encoder and an
              identity-initialised ``Linear(dim -> dim)`` projector per block.
            * cache the VAE latents of the conditioning frame for efficiency.
        """
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                "No Dreamer checkpoint found. Set AIVE_WAN_CKPT_PATH or pass "
                "--dreamer ckpt_path. See dreamer/wan2_2.py for training notes."
            )
        raise NotImplementedError("Wan2_2Dreamer._load_model is not implemented yet.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image_path: str,
        action_sequence: list[str],
        save_dir: str,
        step_idx: int = 0,
    ) -> str | None:
        """Synthesize the imagined next view conditioned on the camera pose.

        The relative pose is derived deterministically from the action
        trajectory (Section 3.1 / Appendix A.2 of the paper):

            ``delta_P = actions_to_relative_pose(action_sequence)``

        TODO(dreamer): implement.
            * compute ``delta_P`` and reshape to the ``(2T, 12)`` conditioning
              layout used by the camera encoder (actual + static reference).
            * run flow-matching sampling conditioned on ``(image, delta_P)``.
            * decode the latent to RGB and save to ``save_dir/imagined_step_{i}.png``.
        """
        relative_pose = actions_to_relative_pose(action_sequence)
        if relative_pose is None:
            return None

        self._load_model()
        raise NotImplementedError(
            "Wan2_2Dreamer.generate is not implemented yet. Train a Dreamer on "
            "the SpaThor Trajectory Records, then implement this method."
        )
