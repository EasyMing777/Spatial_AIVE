"""Abstract Dreamer (world-model) interface for AIVE.

The Dreamer ``W`` is the *generative* component of the framework. Given the
current egocentric observation :math:`i_t` and a planned action trajectory
:math:`\\tau_t`, it synthesizes the imagined future view
:math:`\\tilde{i}_{t+1} = W(i_t, \\tau_t)` (Algorithm 1, line 9 of the paper).

Subclasses only need to implement :meth:`BaseDreamer.generate`. The helper
:func:`actions_to_relative_pose` converts the discrete action vocabulary into
a relative camera pose :math:`\\Delta P_t = P_{t-1}^{-1} P_t \\in SE(3)`, which
is the conditioning signal used by camera-controllable generative backbones
such as Wan2.2-TI2V (see :mod:`dreamer.wan2_2`).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------

#: Canonical action types recognised by the pipeline.
ACTION_TYPES = ("move-forward", "turn-left", "turn-right")


# ---------------------------------------------------------------------------
# Pose arithmetic
# ---------------------------------------------------------------------------


def _rot_y(radians: float) -> np.ndarray:
    """Rotation matrix about the y-axis (right-handed, Y-up)."""
    c, s = math.cos(radians), math.sin(radians)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def actions_to_relative_pose(
    action_sequence: list[str],
    translate_quantum: float = 0.25,
    rotate_quantum: float = 9.0,
) -> np.ndarray | None:
    """Convert a discrete action trajectory into a relative camera pose.

    Each action is expressed as ``"<type> <value>"`` (e.g. ``"move-forward 0.5"``,
    ``"turn-left 18"``). Actions are accumulated into an absolute camera pose
    ``P`` (3 x 4 camera-to-world), and the relative pose between the start and
    the end of the trajectory is returned as :math:`\\Delta P = P_{start}^{-1} P_{end}`.

    Convention
    ----------
    Right-handed, Y-up coordinate frame. The camera looks along the local +Z
    axis; ``move-forward`` advances along the current forward vector,
    ``turn-left`` / ``turn-right`` rotate about the world Y axis by
    ``+/- value`` degrees respectively.

    Parameters
    ----------
    action_sequence : list of str
        Discrete actions, one per element.
    translate_quantum : float
        Translation quantization step used to snap forward distances.
    rotate_quantum : float
        Rotation quantization step used to snap turn angles.

    Returns
    -------
    np.ndarray or None
        A 4 x 4 homogeneous relative camera-to-world transform, or ``None`` if
        the action sequence is empty or malformed.
    """
    if not action_sequence:
        return None

    # Absolute pose accumulator (camera-to-world).
    position = np.zeros(3, dtype=float)
    yaw = 0.0  # radians; 0 = facing +Z

    for action in action_sequence:
        try:
            action_type, value_str = action.split()
            value = float(value_str)
        except (ValueError, AttributeError):
            return None

        if action_type == "move-forward":
            # Snap to the translation grid, then advance along the forward vector.
            value = max(translate_quantum, round(value / translate_quantum) * translate_quantum)
            position += value * np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        elif action_type == "turn-left":
            value = max(rotate_quantum, round(value / rotate_quantum) * rotate_quantum)
            yaw += math.radians(value)
        elif action_type == "turn-right":
            value = max(rotate_quantum, round(value / rotate_quantum) * rotate_quantum)
            yaw -= math.radians(value)
        else:
            return None

    start = np.eye(4)
    end = np.eye(4)
    end[:3, :3] = _rot_y(yaw)
    end[:3, 3] = position

    # Relative transform: world coords of start expressed in end's frame.
    relative = np.linalg.inv(end) @ start
    return relative


# ---------------------------------------------------------------------------
# Abstract Dreamer
# ---------------------------------------------------------------------------


class BaseDreamer(ABC):
    """Abstract world-model interface.

    A concrete Dreamer receives the current view and the planned action
    trajectory, and returns the path to the synthesised next view (or ``None``
    on failure).
    """

    name: str = "base"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        image_path: str,
        action_sequence: list[str],
        save_dir: str,
        step_idx: int = 0,
    ) -> str | None:
        """Synthesize the imagined next view.

        Parameters
        ----------
        image_path : str
            Path to the current observation :math:`i_t`.
        action_sequence : list of str
            Planned discrete actions :math:`\\tau_t` for this step.
        save_dir : str
            Directory in which generated frames should be persisted.
        step_idx : int
            Exploration step index (used to name output files).

        Returns
        -------
        str or None
            Path to the synthesised view :math:`\\tilde{i}_{t+1}`, or ``None``
            if generation failed.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"
