"""Focused tests for the SocialPresenceDetector VLM gate.

Covers 02 Resolved Issue #19 (confirm-anywhere accounting): VLM verification
must run on EVERY pose-positive sampled frame until `min_consistency`
confirmations land, not only the first few — otherwise a clearly-social segment
late in a long, solo-dominated video is starved below the threshold.
"""
import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared import social_presence as sp
from shared.social_presence import SocialPresenceDetector


class _FakeBox:
    """Minimal stand-in for an ultralytics Boxes row."""
    def __init__(self, xyxy, conf):
        self._xyxy, self._conf = xyxy, conf

    @property
    def xyxy(self):
        return np.array([self._xyxy], dtype=float)

    @property
    def conf(self):
        return np.array([self._conf], dtype=float)

    @property
    def id(self):
        return None


class _FakeKeypoints:
    def __init__(self, conf):
        self.conf = conf  # np.ndarray (n_people, 17)


class _FakeResult:
    """One frame of YOLO-pose output with `n_people` confident people in the
    geometric safe zone (not tripping the wearer-limb / chin heuristics)."""
    def __init__(self, n_people=2):
        self.boxes = [_FakeBox([10, 10, 100, 100], 0.9) for _ in range(n_people)]
        self.keypoints = _FakeKeypoints(np.full((n_people, 17), 0.9, dtype=float))


class _FakeCap:
    """cv2.VideoCapture stub yielding `total_frames` blank frames at 30 fps."""
    def __init__(self, total_frames):
        self._total = total_frames
        self._i = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == 5:   # CAP_PROP_FPS
            return 30.0
        if prop == 7:   # CAP_PROP_FRAME_COUNT
            return float(self._total)
        return 0.0

    def read(self):
        if self._i >= self._total:
            return False, None
        self._i += 1
        return True, np.zeros((200, 200, 3), dtype=np.uint8)

    def release(self):
        pass


def _make_detector(monkeypatch, vlm_confirm_sequence):
    """Detector with YOLO + VLM mocked; the VLM confirms per the supplied
    boolean sequence (one entry per pose-positive frame, in order)."""
    det = SocialPresenceDetector(model_path="dummy.pt", vlm_verify=True)

    # Fake YOLO model: .track returns one _FakeResult per input frame; no
    # `.predictor` attr so the tracker-reset path is a no-op.
    class _FakeModel:
        def track(self, frames, **kwargs):
            return [_FakeResult(n_people=2) for _ in frames]
    det._model = _FakeModel()

    calls = {"n": 0}
    seq = list(vlm_confirm_sequence)

    def fake_confirm(frame_bgr):
        i = calls["n"]
        calls["n"] += 1
        return seq[i] if i < len(seq) else False
    monkeypatch.setattr(det, "_vlm_confirms_multiple_people", fake_confirm)
    # Never let the stereo / chin gates fire for this geometry/aspect ratio.
    monkeypatch.setattr(det, "_vlm_confirms_side_by_side_stereo", lambda f: False)
    monkeypatch.setattr(det, "_vlm_face_oriented_upward", lambda f, b: False)

    # 700 frames @ 30fps, sampled at 1/3 fps -> interval 90 -> ~8 sampled frames.
    monkeypatch.setattr(sp.cv2, "VideoCapture", lambda *a, **k: _FakeCap(700))
    return det, calls


class _ExistsPath:
    """Path-like whose .exists() is True and .name is set (detect() needs both)."""
    name = "long_video.mp4"
    def exists(self):
        return True
    def __fspath__(self):
        return "long_video.mp4"


def test_confirm_anywhere_passes_late_social_segment(monkeypatch):
    """The starvation case: the first 5 pose-positive frames are VLM-negative,
    the social segment confirms only on frames 6 and 7. The old 5-attempt cap
    rejected this video; confirm-anywhere must now reach min_consistency=2."""
    det, calls = _make_detector(monkeypatch, [False] * 5 + [True, True, False])
    result = det.detect(_ExistsPath(), fast_mode=True, min_consistency=2)
    assert result is True
    # It kept verifying past the old 5-frame cap to find the late confirmations.
    assert calls["n"] >= 7


def test_genuinely_solo_video_still_rejected(monkeypatch):
    """Confirm-anywhere must not weaken the gate: a video the VLM never confirms
    (pose double-counts the wearer, say) stays rejected in fast_mode."""
    det, calls = _make_detector(monkeypatch, [False] * 8)
    result = det.detect(_ExistsPath(), fast_mode=True, min_consistency=2)
    assert result is False
