"""Unit tests for the Dreamer interface, pose conversion, and MockDreamer."""

import os

import cv2
import numpy as np
import pytest

from dreamer.base import actions_to_relative_pose
from dreamer.mock import MockDreamer
from dreamer.wan2_2 import Wan2_2Dreamer

# ---------------------------------------------------------------------------
# actions_to_relative_pose
# ---------------------------------------------------------------------------


def test_empty_sequence_is_none():
    assert actions_to_relative_pose([]) is None


def test_malformed_sequence_is_none():
    assert actions_to_relative_pose(["not-an-action"]) is None


def test_forward_moves_along_z():
    pose = actions_to_relative_pose(["move-forward 0.5"])
    assert pose is not None and pose.shape == (4, 4)
    # forward translation must displace the camera along its local Z axis
    assert abs(pose[2, 3]) > 0


def test_turn_produces_rotation():
    pose = actions_to_relative_pose(["turn-left 90"])
    assert pose is not None
    # a 90-degree rotation about Y: cos(90) ~ 0 on the diagonal
    assert abs(pose[0, 0]) < 1e-6
    assert abs(pose[0, 2]) - 1 < 1e-6


def test_values_snap_to_grid():
    # 0.37 m snaps down to the 0.25 m grid
    pose = actions_to_relative_pose(["move-forward 0.37"])
    assert pose is not None
    assert abs(pose[2, 3]) == pytest.approx(0.25, abs=1e-6)


def test_mixed_sequence_is_sane():
    pose = actions_to_relative_pose(["turn-left 18", "move-forward 0.5", "turn-right 9"])
    assert pose is not None
    assert np.allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)  # pure rotation part


# ---------------------------------------------------------------------------
# MockDreamer
# ---------------------------------------------------------------------------


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        MockDreamer(mode="bogus")


def test_identity_returns_copy(tmp_path):
    img_path = tmp_path / "in.png"
    cv2.imwrite(str(img_path), np.full((32, 32, 3), 120, dtype=np.uint8))

    dreamer = MockDreamer(mode="identity")
    out = dreamer.generate(str(img_path), ["move-forward 0.5"], str(tmp_path / "step"), step_idx=0)

    assert out is not None and os.path.exists(out)
    assert cv2.imread(out).shape == (32, 32, 3)


def test_shift_changes_content(tmp_path):
    in_path = tmp_path / "in.png"
    cv2.imwrite(str(in_path), np.full((64, 64, 3), 120, dtype=np.uint8))

    dreamer = MockDreamer(mode="shift")
    out = dreamer.generate(str(in_path), ["turn-left 9"], str(tmp_path / "step"), step_idx=0)
    assert out is not None and os.path.exists(out)


def test_missing_input_returns_none(tmp_path):
    dreamer = MockDreamer(mode="identity")
    assert (
        dreamer.generate(str(tmp_path / "nope.png"), ["move-forward 0.5"], str(tmp_path), 0) is None
    )


# ---------------------------------------------------------------------------
# Wan2_2Dreamer (extension point)
# ---------------------------------------------------------------------------


def test_wan2_2_requires_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("AIVE_WAN_CKPT_PATH", raising=False)
    dreamer = Wan2_2Dreamer()  # no checkpoint configured
    with pytest.raises(FileNotFoundError):
        dreamer.generate("unused.png", ["move-forward 0.5"], str(tmp_path), 0)
