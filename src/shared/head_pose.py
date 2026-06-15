"""Shared head-pose estimator (docs/03e Issue 1, Option A).

03e's nod/shake premise needs true HEAD ORIENTATION, but 03a only had L2CS gaze
(where the eyes point), which moves with saccades while the head is still. This
module derives head Euler pitch/yaw from MediaPipe FaceLandmarker's
`facial_transformation_matrixes` so 03a can emit `head_pitch_rad`/`head_yaw_rad`
alongside gaze, and 03e can consume head pose instead of gaze.

Design notes:
  - Lazy MediaPipe Tasks FaceLandmarker (IMAGE mode), mirroring 03a's BlazeFace
    gate — so it is created per-process and is safe under the macOS 'spawn'
    parallel workers (never pickled).
  - `estimate()` returns None when no face landmarks are found. FaceLandmarker
    has its OWN tracking-loss on small / steeply-angled bystander faces, and the
    caller MUST treat None as missing data (NaN), exactly like the NoFace gaze
    gate — never as a real 0.0 reading (see docs/03e Resolved #3).
  - Pitch is rotation about the camera X-axis (nod), yaw about Y (shake), to
    match 03e's axis assumption. Returned in radians.
"""
import math
from pathlib import Path

import numpy as np

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "mediapipe" / "face_landmarker.task"
)


class HeadPoseEstimator:
    """Face crop (BGR) -> (head_pitch_rad, head_yaw_rad), or None when no face."""

    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._landmarker = None
        # Available only if mediapipe is importable AND the model asset is present;
        # otherwise the caller silently falls back to gaze (no head_* emitted).
        self.available = MP_AVAILABLE and self.model_path.exists()

    @property
    def landmarker(self):
        if self._landmarker is None:
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                output_facial_transformation_matrixes=True,
                num_faces=1,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    @staticmethod
    def _pitch_yaw_from_matrix(matrix):
        """Decompose the 4x4 face->camera transform to (pitch, yaw) in radians.

        Standard XYZ Euler extraction from the top-left 3x3 rotation: pitch is
        rotation about X (vertical head nod), yaw about Y (horizontal shake).
        Roll (about Z) is discarded — 03e only needs pitch/yaw.
        """
        r = np.asarray(matrix, dtype=float)[:3, :3]
        sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        return float(pitch), float(yaw)

    def estimate(self, crop_bgr):
        """Return (pitch_rad, yaw_rad), or None if unavailable / no face found."""
        if not self.available or crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return None
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.facial_transformation_matrixes:
                return None
            return self._pitch_yaw_from_matrix(result.facial_transformation_matrixes[0])
        except Exception:
            # Head pose is best-effort; a failure must never break the caller's
            # hot loop — fall back to "no head pose for this sample".
            return None

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
