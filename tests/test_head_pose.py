"""Tests for the shared head-pose estimator (docs/03e Issue 1, Option A)."""
import math

import numpy as np

from src.shared.head_pose import HeadPoseEstimator


def _rot_x(deg):
    a = math.radians(deg)
    M = np.eye(4)
    M[1, 1], M[1, 2] = math.cos(a), -math.sin(a)
    M[2, 1], M[2, 2] = math.sin(a), math.cos(a)
    return M


def _rot_y(deg):
    a = math.radians(deg)
    M = np.eye(4)
    M[0, 0], M[0, 2] = math.cos(a), math.sin(a)
    M[2, 0], M[2, 2] = -math.sin(a), math.cos(a)
    return M


def test_identity_is_zero_pose():
    p, y = HeadPoseEstimator._pitch_yaw_from_matrix(np.eye(4))
    assert abs(p) < 1e-9 and abs(y) < 1e-9


def test_pitch_axis_is_nod():
    # Rotation about X (head nodding down/up) shows up as pitch, not yaw.
    p, y = HeadPoseEstimator._pitch_yaw_from_matrix(_rot_x(20))
    assert abs(math.degrees(p) - 20) < 1e-3
    assert abs(y) < 1e-6


def test_yaw_axis_is_shake():
    # Rotation about Y (head shaking left/right) shows up as yaw, not pitch.
    p, y = HeadPoseEstimator._pitch_yaw_from_matrix(_rot_y(30))
    assert abs(math.degrees(y) - 30) < 1e-3
    assert abs(p) < 1e-6


def test_unavailable_when_asset_missing():
    hp = HeadPoseEstimator(model_path="/nonexistent/face_landmarker.task")
    assert hp.available is False
    # estimate must degrade to None (never raise) so the 03a hot loop is safe.
    assert hp.estimate(None) is None
    assert hp.estimate(np.zeros((0, 0, 3), dtype=np.uint8)) is None
