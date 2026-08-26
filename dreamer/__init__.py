"""World-model (Dreamer) package for AIVE.

The Dreamer synthesizes the imagined future view given the current observation
and a planned action trajectory. This package ships:

* :class:`dreamer.base.BaseDreamer` — the abstract interface;
* :class:`dreamer.mock.MockDreamer` — a deterministic placeholder for testing;
* :class:`dreamer.wan2_2.Wan2_2Dreamer` — the Wan2.2-TI2V extension point.

Use :func:`create_dreamer` to instantiate a backend from a ``--dreamer_type``
string (``mock`` | ``wan2_2``).
"""

from __future__ import annotations

from typing import Any

from dreamer.base import BaseDreamer, actions_to_relative_pose
from dreamer.mock import MockDreamer
from dreamer.wan2_2 import Wan2_2Dreamer

__all__ = [
    "BaseDreamer",
    "MockDreamer",
    "Wan2_2Dreamer",
    "actions_to_relative_pose",
    "create_dreamer",
]


def create_dreamer(dreamer_type: str, **kwargs: Any) -> BaseDreamer:
    """Factory for Dreamer backends.

    Parameters
    ----------
    dreamer_type : str
        One of ``"mock"`` or ``"wan2_2"``.
    **kwargs
        Backend-specific configuration (e.g. ``mode`` for the mock,
        ``ckpt_path`` for Wan2.2).

    Returns
    -------
    BaseDreamer
        An instantiated Dreamer.
    """
    if dreamer_type == "mock":
        return MockDreamer(**kwargs)
    if dreamer_type == "wan2_2":
        return Wan2_2Dreamer(**kwargs)
    raise ValueError(f"Unknown dreamer_type {dreamer_type!r}; expected 'mock' or 'wan2_2'.")
