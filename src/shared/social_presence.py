import os
import cv2
import gc
import tempfile
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO
from pathlib import Path

try:
    from src.models_config import get_model
except ImportError:
    from models_config import get_model

# COCO-17 keypoint indices used by ultralytics YOLO-pose. We gate "real
# bystander" on visible head (any of nose/eyes/ears) AND at least one
# shoulder, which is the cheapest geometric proxy for a torso-attached
# human and intrinsically rejects disconnected limbs / equipment that the
# bbox-only yolov8n flagged as Resolved Issue #22.
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KEYPOINT_CONF_THRESHOLD = 0.3
MAX_VLM_VERIFY_FRAMES = 5


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_model_path(path_str: str) -> Path:
    """Registry entries for on-disk model bundles are stored as repo-relative
    paths; resolve to absolute so MediaPipe's `model_asset_path` is robust to
    the caller's CWD."""
    p = Path(path_str)
    return p if p.is_absolute() else (_PROJECT_ROOT / p)


class SocialPresenceDetector:
    def __init__(self, model_path=None, vlm_verify=None, vlm_model=None):
        # Model identifier resolution priority: explicit constructor arg ->
        # tier-per-host registry (`social_presence_pose`). The registry's
        # default is yolov8n-pose.pt on every tier today; future tier
        # variations (e.g. yolov8s-pose at `large`) only need editing
        # src/models_config.py.
        self.model_path = model_path or get_model("social_presence_pose")
        self._model = None
        self._hand_landmarker = None
        self._hand_landmarker_path = _resolve_model_path(get_model("social_presence_hand_landmarker"))

        if vlm_verify is None:
            vlm_verify = os.getenv("SAF_VLM_VERIFY_SOCIAL", "").lower() in ("1", "true", "yes")
        self.vlm_verify = vlm_verify
        # SAF_VLM_VERIFY_MODEL retained as the per-detector override; falls
        # back to the tier-per-host registry's `social_presence_vlm_verify`
        # entry (defaults to `moondream` on small/medium tiers).
        self.vlm_model = vlm_model or os.getenv(
            "SAF_VLM_VERIFY_MODEL", get_model("social_presence_vlm_verify")
        )
        self._ollama = None

    @property
    def model(self):
        if self._model is None:
            # Lazy loading to save memory if not used
            print(f"[SocialPresenceDetector] Loading YOLO model: {self.model_path}")
            self._model = YOLO(self.model_path)
        return self._model

    @property
    def hand_landmarker(self):
        # MediaPipe Tasks API HandLandmarker (Resolved Issue #9). The legacy
        # `mp.solutions.hands` namespace is gone in mediapipe>=0.10.30, so
        # this is the only forward-compatible entry point. RunningMode.IMAGE
        # avoids the monotonic-timestamp constraint of VIDEO mode — the
        # bbox-only consumer here doesn't benefit from cross-frame tracking
        # smoothing, and IMAGE mode lets the same detector be reused across
        # videos without resetting timestamps.
        if self._hand_landmarker is None:
            print(f"[SocialPresenceDetector] Loading MediaPipe HandLandmarker: {self._hand_landmarker_path}")
            base_opts = mp_python.BaseOptions(model_asset_path=str(self._hand_landmarker_path))
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
            self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        return self._hand_landmarker

    @property
    def ollama_client(self):
        if self._ollama is None:
            import ollama
            self._ollama = ollama
        return self._ollama

    def unload(self):
        """ Explicitly unload the model and clear memory """
        if self._model is not None:
            print(f"[SocialPresenceDetector] Unloading YOLO model...")
            del self._model
            self._model = None

        if self._hand_landmarker is not None:
            print(f"[SocialPresenceDetector] Unloading MediaPipe HandLandmarker...")
            self._hand_landmarker.close()
            self._hand_landmarker = None

        # Force garbage collection and clear MPS/CUDA cache
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _has_head_and_shoulder(self, kps_conf):
        """ Return True if at least one head keypoint AND at least one
        shoulder keypoint clear the confidence threshold. """
        head_visible = any(
            kps_conf[i] >= KEYPOINT_CONF_THRESHOLD
            for i in (KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR)
        )
        shoulder_visible = (
            kps_conf[KP_LEFT_SHOULDER] >= KEYPOINT_CONF_THRESHOLD
            or kps_conf[KP_RIGHT_SHOULDER] >= KEYPOINT_CONF_THRESHOLD
        )
        return head_visible and shoulder_visible

    def _vlm_ask_yes_no(self, frame_bgr, prompt: str, default_on_error: bool) -> bool:
        """ Send a single YES/NO image+prompt request to the configured fast
        VLM. Returns True iff the response starts with "YES"; on
        infrastructure errors returns `default_on_error` so the caller can
        choose whether a VLM outage is a soft-pass or a soft-fail. """
        try:
            h, w = frame_bgr.shape[:2]
            max_dim = 768
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame_bgr = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)

            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                cv2.imwrite(tmp_path, frame_bgr)
                response = self.ollama_client.chat(
                    model=self.vlm_model,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [tmp_path],
                    }],
                )
                content = response['message']['content'].strip().upper()
                return content.startswith("YES")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except Exception as e:
            print(f"[SocialPresenceDetector] VLM verification failed: {e}")
            return default_on_error

    def _vlm_confirms_multiple_people(self, frame_bgr) -> bool:
        """ Ask the configured fast VLM whether the frame clearly shows two
        or more distinct people (excluding the wearer's own limbs).
        Returns True if the VLM confirms, False if it explicitly denies, and
        True on infrastructure errors so VLM availability is not the sole
        reason to drop an otherwise-valid YOLO-pose-confirmed frame. """
        return self._vlm_ask_yes_no(
            frame_bgr,
            (
                "Does this image clearly show two or more real, physical people "
                "actually present in the scene? "
                "Do NOT count people shown on TVs, monitors, phone screens, "
                "photographs, posters, paintings, magazines, or reflections in mirrors. "
                "Do NOT count the camera operator's own hands or limbs. "
                "Respond with just YES or NO."
            ),
            default_on_error=True,
        )

    def _vlm_face_oriented_upward(self, frame_bgr, bbox) -> bool:
        """ Per Resolved Issue #11, the camera wearer's own chin / lower face
        sometimes peeks into the bottom edge of an egocentric capture and
        gets detected as a separate person whose head keypoints (mouth, jaw)
        are all visible. The geometric Anti-Wearer Heuristic misses this
        because the bbox doesn't extend to the literal bottom pixel row.
        We crop the candidate bbox (with a small pad for context) and ask
        the VLM whether the face is angled upward — the signature of looking
        up at the wearer's own face from below.
        Returns False on VLM errors so a single outage doesn't drop every
        bottom-edge bystander.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        pad_x = int(0.05 * w)
        pad_y = int(0.05 * h)
        x1c = max(0, x1 - pad_x)
        x2c = min(w, x2 + pad_x)
        y1c = max(0, y1 - pad_y)
        y2c = min(h, y2 + pad_y)
        crop = frame_bgr[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return False
        return self._vlm_ask_yes_no(
            crop,
            (
                "Is this person's face oriented upward toward the camera, as if "
                "you are looking up at them from below? This pose is the "
                "signature of an egocentric camera operator's own chin or lower "
                "face peeking into the frame from underneath. "
                "Respond with just YES or NO."
            ),
            default_on_error=False,
        )

    def _vlm_confirms_side_by_side_stereo(self, frame_bgr) -> bool:
        """ Per Resolved Issue #10, ego4d capture rigs occasionally emit a
        side-by-side stereoscopic image where the left and right halves render
        the same scene. YOLO-pose then double-counts the wearer's own limbs as
        two bystanders. Detect with a VLM YES/NO so we don't need to encode an
        aspect-ratio heuristic that misclassifies legitimate ultrawide
        captures. Defaults to False on VLM errors — a stereo positive is the
        rare class, so a missed-stereo from an outage is far less costly than
        false-rejecting every frame.
        """
        return self._vlm_ask_yes_no(
            frame_bgr,
            (
                "Is this image a side-by-side stereoscopic capture where the left "
                "half and the right half show the same scene rendered twice "
                "(as if from a stereo camera rig)? "
                "Respond with just YES or NO."
            ),
            default_on_error=False,
        )

    def detect(self, video_path: Path, sample_rate_fps=1/3.0, fast_mode=False, min_consistency=2, return_hands=False):
        """
        Detect persons in a video.

        Args:
            video_path: Path to the video file.
            sample_rate_fps: How many frames to sample per second. Default is
                1/3 (one frame every 3 seconds) per Resolved Issue #11 — at 1
                FPS the per-video VLM budget was dominated by the bottom-edge
                wearer-chin re-checks; sampling at 1/3 FPS reclaims that
                budget while still hitting `min_consistency` on real
                multi-person videos.
            fast_mode: If True, returns True as soon as 'min_consistency' person frames are detected.
                      If False, returns a list of all detections per frame.
            min_consistency: Number of frames with social presence required to confirm (default 2).
            return_hands: If True, also returns MediaPipe Hands boxes per frame (consumed by
                          Layer 03a as documented `hand_detections` per Resolved Issue #17).
        """
        if not video_path.exists():
            return False if fast_mode else ([] if not return_hands else ([], []))

        # Reset ByteTrack state to prevent ID/trajectory bleeding across videos.
        # The model uses persist=True for within-video temporal consistency, so
        # we must explicitly clear trackers between videos.
        try:
            predictor = getattr(self.model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) if predictor is not None else None
            if trackers:
                for tracker in trackers:
                    tracker.reset()
        except Exception as e:
            print(f"[SocialPresenceDetector] Warning: Could not reset tracker state: {e}")

        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                return False if fast_mode else ([] if not return_hands else ([], []))

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps == 0 or total_frames == 0:
                return False if fast_mode else ([] if not return_hands else ([], []))

            frame_interval = int(max(1, fps / sample_rate_fps))

            all_detections = []
            all_hands = []
            detected_frames_count = 0
            vlm_verified_count = 0
            vlm_attempts = 0
            stereo_checked = False

            batch_frames = []
            batch_timestamps = []
            BATCH_SIZE = 16  # Adjustable batch size based on memory

            # Use sequential reading instead of seeking for stability on macOS
            current_frame_idx = 0
            while True:
                ret, frame = cap.read()

                # Only process frames at the specified interval
                if ret and current_frame_idx % frame_interval == 0:
                    if frame is not None and frame.size > 0:
                        batch_frames.append(frame)
                        batch_timestamps.append(current_frame_idx / fps)

                current_frame_idx += 1
                is_end = not ret or current_frame_idx >= total_frames

                # Process batch if full or at end of video
                if len(batch_frames) >= BATCH_SIZE or (is_end and len(batch_frames) > 0):
                    # Use YOLO internal batching with ByteTrack
                    results = self.model.track(batch_frames, classes=[0], verbose=False, conf=0.5, batch=len(batch_frames), persist=True, tracker="bytetrack.yaml")

                    for i, result in enumerate(results):
                        timestamp = batch_timestamps[i]
                        frame_detections = []
                        has_bystander_in_frame = False
                        img_h, img_w = batch_frames[i].shape[:2]

                        # MediaPipe Hand Detection is only needed when the caller
                        # asks for `return_hands` (i.e. Node 02 building the
                        # `hand_detections` schema field for Layer 03a). YOLO-pose
                        # keypoint validation below makes hand-overlap suppression
                        # redundant, so skipping MediaPipe entirely on the Node 01
                        # streaming filter is a real per-batch perf win.
                        hand_boxes = []
                        if return_hands:
                            # Tasks API: wrap the BGR frame as `mp.Image` in
                            # SRGB layout (i.e. RGB byte order). The result's
                            # `hand_landmarks` is a list-of-lists of normalized
                            # landmarks — each landmark has `.x`/`.y` in [0, 1]
                            # relative to the input image, matching the legacy
                            # solutions API's coordinate shape.
                            frame_rgb = cv2.cvtColor(batch_frames[i], cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                            hand_results = self.hand_landmarker.detect(mp_image)
                            for hand_landmarks in hand_results.hand_landmarks:
                                x_min, y_min = img_w, img_h
                                x_max, y_max = 0, 0
                                for lm in hand_landmarks:
                                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                                    x_min, y_min = min(x_min, x), min(y_min, y)
                                    x_max, y_max = max(x_max, x), max(y_max, y)
                                pad = 20
                                hand_boxes.append((max(0, x_min - pad), max(0, y_min - pad), min(img_w, x_max + pad), min(img_h, y_max + pad)))

                        # YOLO-pose keypoint tensors. The boxes and keypoints arrays
                        # are co-indexed: result.boxes[k] corresponds to result.keypoints[k].
                        kps_conf = None
                        keypoints_obj = getattr(result, "keypoints", None)
                        if keypoints_obj is not None and getattr(keypoints_obj, "conf", None) is not None:
                            kps_conf_tensor = keypoints_obj.conf
                            kps_conf = kps_conf_tensor.cpu().numpy() if hasattr(kps_conf_tensor, "cpu") else kps_conf_tensor

                        for box_idx, box in enumerate(result.boxes):
                            coords = [int(v) for v in box.xyxy[0].tolist()]
                            x1, y1, x2, y2 = coords

                            # Refined Anti-Wearer Heuristic (kept as a cheap geometric
                            # prefilter on top of the pose-keypoint check below)
                            is_limb = (y2 > 0.95 * img_h) and (y1 > 0.15 * img_h)
                            is_ghost = (y1 == 0) and (y2 > 0.90 * img_h)

                            if is_limb or is_ghost:
                                continue

                            # YOLO-pose head+shoulder gate (Option A from Resolved Issue #22).
                            # Disconnected limbs / mannequins / equipment never satisfy both
                            # a visible head keypoint and a visible shoulder keypoint.
                            if kps_conf is not None and box_idx < len(kps_conf):
                                if not self._has_head_and_shoulder(kps_conf[box_idx]):
                                    continue
                            else:
                                # Pose model returned no keypoints for this detection:
                                # treat as a non-confirmable bystander and skip.
                                continue

                            # Gaze-direction wearer-chin gate (Resolved Issue #11).
                            # When a detection sits in the bottom 20% of the
                            # frame, ask the VLM whether the face is angled
                            # upward — the signature of the wearer's own face
                            # peeking into the frame from below. Skip the
                            # detection on YES.
                            if self.vlm_verify and y2 > 0.80 * img_h:
                                if self._vlm_face_oriented_upward(batch_frames[i], coords):
                                    continue

                            # Extract tracking ID if available
                            person_id = int(box.id[0]) if box.id is not None else len(frame_detections)

                            frame_detections.append({
                                "person_id": person_id,
                                "timestamp_sec": timestamp,
                                "bounding_box": coords,
                                "confidence": float(box.conf[0])
                            })
                            has_bystander_in_frame = True

                        if has_bystander_in_frame:
                            detected_frames_count += 1

                            # Side-by-side stereo VLM gate (Resolved Issue #10):
                            # Stereo format is a video-wide property of the
                            # capture rig, so we ask Moondream once on the
                            # first pose-positive frame. A YES means YOLO is
                            # double-counting the wearer's own limbs across
                            # the left/right halves — drop the whole video.
                            if self.vlm_verify and not stereo_checked:
                                stereo_checked = True
                                if self._vlm_confirms_side_by_side_stereo(batch_frames[i]):
                                    print(
                                        f"[SocialPresenceDetector] Stereo gate rejected {video_path.name}: "
                                        f"side-by-side stereoscopic capture detected"
                                    )
                                    if fast_mode:
                                        return False
                                    return ([], []) if return_hands else []

                            # VLM Early-Exit Verification (Option B from Resolved Issue #22):
                            # Verify up to MAX_VLM_VERIFY_FRAMES candidate frames against a
                            # fast VLM. Once `min_consistency` frames are VLM-confirmed, the
                            # video has passed the gate and we stop spending VLM budget.
                            if (
                                self.vlm_verify
                                and vlm_verified_count < min_consistency
                                and vlm_attempts < MAX_VLM_VERIFY_FRAMES
                            ):
                                vlm_attempts += 1
                                if self._vlm_confirms_multiple_people(batch_frames[i]):
                                    vlm_verified_count += 1

                        if frame_detections or (return_hands and hand_boxes):
                            all_detections.append(frame_detections)
                            all_hands.append({
                                "timestamp_sec": timestamp,
                                "hand_boxes": hand_boxes
                            })

                    batch_frames = []
                    batch_timestamps = []

                    # Early-exit in fast_mode: gated on VLM verification when enabled,
                    # otherwise on raw YOLO-pose frame count as before.
                    if fast_mode:
                        if self.vlm_verify:
                            if vlm_verified_count >= min_consistency:
                                return True
                        elif detected_frames_count >= min_consistency:
                            return True

                if is_end:
                    break

            # Video-level VLM gate: if VLM verification was requested but no
            # candidate frame ever cleared the confirmation threshold, drop
            # the whole video. This is the "two-pass" verification the
            # remediation path called for — YOLO-pose proposes, VLM disposes.
            if self.vlm_verify and detected_frames_count > 0 and vlm_verified_count < min_consistency:
                print(
                    f"[SocialPresenceDetector] VLM gate rejected {video_path.name}: "
                    f"{vlm_verified_count}/{min_consistency} confirmed across {vlm_attempts} attempts"
                )
                if fast_mode:
                    return False
                return ([], []) if return_hands else []

            if fast_mode:
                return detected_frames_count >= min_consistency
            if return_hands:
                return all_detections, all_hands
            return all_detections
        finally:
            cap.release()
