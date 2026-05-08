import os
import json
import traceback
import cv2
import numpy as np
from pathlib import Path

# Fallback: using ultralytics YOLO pose instead of MMPose
try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class MotorResonancePipeline:
    # --- Detection Tuning ---
    CHAOS_FLOOR = 3.0
    CHAOS_NORMALIZER = 20.0
    SPIKE_RELATIVE_THRESHOLD = 0.7
    FRAME_RESIZE_FACTOR = 0.5
    TARGET_FLOW_FPS = 10.0
    KPT_CONFIDENCE_THRESHOLD = 0.5
    VELOCITY_NORMALIZER = 0.5
    VELOCITY_CAP = 10.0
    RESONANCE_WINDOW_SEC = 0.5
    EMPATHY_NORMALIZER = 5.0
    MIRRORING_VFLOW_THRESHOLD = -1.0
    MIRRORING_TIME_WINDOW_SEC = 0.5
    MIRRORING_DELTA_THRESHOLD = 0.1
    MIRRORING_SCALAR_NORMALIZER = 0.5
    PREV_KPTS_CARRY_FORWARD_SEC = 0.5
    RELEVANT_KPT_INDICES = (5, 6, 9, 10, 11, 12)

    def __init__(self, input_manifest_path, output_result_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03f_motor_resonance_errors.json"
        self.force = force
        self.model = None

        self.processed_ids = set()
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting fresh.")

        self._init_model()

    def _init_model(self):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics is required for the Motor Resonance layer fallback. "
                               "Please install it via 'pip install ultralytics torch torchvision'.")

        try:
            device = 'cpu'
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'

            print(f"Initializing YOLOv8 Pose (fallback for MMPose) on {device}...")
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_path = base_dir / "yolov8n-pose.pt"

            if not model_path.exists():
                print(f"Warning: Model {model_path} not found. Attempting to download via ultralytics...")

            self.model = YOLO(str(model_path) if model_path.exists() else "yolov8n-pose.pt")
            self.device = device

        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLOv8 Pose: {e}")

    def _log_error(self, video_id, error):
        error_entry = {
            "video_id": video_id,
            "error": str(error),
            "traceback": traceback.format_exc()
        }
        print(f"Error processing {video_id}: {error}")

        errors = []
        if self.error_log_path.exists():
            try:
                with open(self.error_log_path, 'r') as f:
                    errors = json.load(f)
            except Exception:
                pass

        errors.append(error_entry)

        temp_err = self.error_log_path.with_suffix('.tmp')
        with open(temp_err, 'w') as f:
            json.dump(errors, f, indent=4)
        temp_err.replace(self.error_log_path)

    def _sentinel(self, video_id, reason):
        return {
            "video_id": video_id,
            "layer": "03f_motor_resonance",
            "tasks_analyzed": [],
            "skipped_reason": reason,
        }

    def run(self):
        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)

        results = []
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    results = json.load(f)
            except Exception:
                pass

        for entry in registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue

            print(f"Processing Motor Resonance for video: {video_id}")
            try:
                result = self.process_video(entry)
                results.append(result)
                self.processed_ids.add(video_id)

                # Atomic write
                temp_out = self.output_result_path.with_suffix('.tmp')
                with open(temp_out, 'w') as f:
                    json.dump(results, f, indent=4)
                temp_out.replace(self.output_result_path)
            except Exception as e:
                self._log_error(video_id, e)

        print(f"Final count: {len(results)} videos processed for Motor Resonance.")

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['video_path'])

        if not video_path.exists():
            print(f"File not found: {video_path}")
            return self._sentinel(video_id, "file_not_found")

        bystanders = entry.get('bystander_detections', [])
        tasks = entry.get('identified_tasks', [])

        if not bystanders or not tasks:
            print(f"No bystanders or tasks found for {video_id}.")
            return self._sentinel(video_id, "no_bystanders_or_tasks")

        any_ego_spikes = False
        any_pose_data = False
        tasks_analyzed = []
        for task in tasks:
            task_id = task.get('task_id', 'unknown')
            meta = task.get('task_temporal_metadata', {})
            reaction_window = meta.get('task_reaction_window_sec')

            if not reaction_window or len(reaction_window) != 2:
                continue

            start_sec, end_sec = reaction_window

            # Step 1: EgoMotion Extraction (returns chaos spikes for flinch
            # correlation AND a separate vertical-flow timeline that mirroring
            # consumes independent of the chaos floor — see Resolved Issue 16).
            ego_spikes, vertical_flow_timeline, max_chaos_score = self._extract_ego_motion(
                video_path, start_sec, end_sec, bystanders
            )

            if ego_spikes:
                any_ego_spikes = True
            elif not vertical_flow_timeline:
                continue

            per_person = []
            for bystander in bystanders:
                person_id = bystander.get('person_id')
                timestamps_sec = bystander.get('timestamps_sec', [])
                bounding_boxes = bystander.get('bounding_boxes', [])

                if not timestamps_sec or not bounding_boxes:
                    continue

                # Step 2 & 3: Pose Extraction & Correlating Resonance
                pose_analysis = self._extract_and_correlate_pose(
                    video_path, timestamps_sec, bounding_boxes,
                    start_sec, end_sec, ego_spikes, vertical_flow_timeline,
                )

                if pose_analysis:
                    any_pose_data = True
                    per_person.append({
                        "person_id": person_id,
                        "bystander_pose_velocity_peak": pose_analysis['velocity_peak'],
                        "resonance_delay_sec": pose_analysis['delay_sec'],
                        "motor_resonance_detected": pose_analysis['resonance_detected'],
                        "empathy_scalar": pose_analysis['empathy_scalar'],
                        "mirroring_detected": pose_analysis.get('mirroring_detected', False),
                        "mirroring_scalar": pose_analysis.get('mirroring_scalar', 0.0)
                    })

            if per_person:
                tasks_analyzed.append({
                    "task_id": task_id,
                    "ego_kinetic_chaos_score": round(max_chaos_score, 2),
                    "per_person": per_person
                })

        if not tasks_analyzed:
            if not any_ego_spikes:
                return self._sentinel(video_id, "no_ego_spikes")
            if not any_pose_data:
                return self._sentinel(video_id, "no_pose_data")
            return self._sentinel(video_id, "no_pose_data")

        return {
            "video_id": video_id,
            "layer": "03f_motor_resonance",
            "tasks_analyzed": tasks_analyzed
        }

    def _bystander_mask_for_frame(self, t, frame_shape_ds, bystanders):
        """Build a boolean mask over the downsampled frame zeroing out the
        nearest-timestamp bbox for every bystander. ``True`` = keep (background)."""
        h_ds, w_ds = frame_shape_ds
        keep = np.ones((h_ds, w_ds), dtype=bool)
        scale = self.FRAME_RESIZE_FACTOR
        for b in bystanders:
            ts = b.get('timestamps_sec', [])
            bxs = b.get('bounding_boxes', [])
            if not ts or not bxs:
                continue
            # Nearest timestamp lookup
            arr = np.asarray(ts, dtype=float)
            j = int(np.argmin(np.abs(arr - t)))
            if j >= len(bxs):
                continue
            x1, y1, x2, y2 = bxs[j]
            x1 = max(0, int(x1 * scale))
            y1 = max(0, int(y1 * scale))
            x2 = min(w_ds, int(x2 * scale))
            y2 = min(h_ds, int(y2 * scale))
            if x2 > x1 and y2 > y1:
                keep[y1:y2, x1:x2] = False
        return keep

    def _extract_ego_motion(self, video_path, start_sec, end_sec, bystanders):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], [], 0.0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                return [], [], 0.0

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            frame_stride = max(1, int(round(fps / self.TARGET_FLOW_FPS)))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            ret, prev_frame = cap.read()
            if not ret:
                return [], [], 0.0

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.resize(prev_gray, (0, 0),
                                   fx=self.FRAME_RESIZE_FACTOR,
                                   fy=self.FRAME_RESIZE_FACTOR)

            chaos_scores = []
            current_frame_idx = start_frame + frame_stride

            while current_frame_idx <= end_frame:
                if frame_stride > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (0, 0),
                                  fx=self.FRAME_RESIZE_FACTOR,
                                  fy=self.FRAME_RESIZE_FACTOR)

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # 95th-percentile flow magnitude — left in raw flow-magnitude
                # units regardless of stride. The chaos floor and relative
                # spike threshold were calibrated against per-frame-pair flow
                # magnitudes, and impulsive jolts (random pixel content
                # between two frames) do not scale with stride the way
                # continuous motion does. Stride-normalizing here would have
                # erased detectable jolts in the test fixture.
                chaos_score = float(np.percentile(mag, 95))

                # Mask out bystander regions before averaging vertical flow so
                # the mirroring signal reflects *background* (camera) motion
                # only, per the documented intent (Resolved Issue 12). For
                # continuous camera motion the mean displacement between two
                # stride-spaced frames is ~stride× the per-frame rate, so
                # divide by stride to keep MIRRORING_VFLOW_THRESHOLD calibrated
                # in the same units as before.
                mask = self._bystander_mask_for_frame(
                    current_frame_idx / fps, gray.shape, bystanders,
                )
                if mask.any():
                    mean_v = float(np.mean(flow[..., 1][mask])) / float(frame_stride)
                else:
                    mean_v = float(np.mean(flow[..., 1])) / float(frame_stride)

                timestamp = current_frame_idx / fps
                chaos_scores.append((timestamp, chaos_score, mean_v))

                prev_gray = gray
                current_frame_idx += frame_stride
        finally:
            cap.release()

        if not chaos_scores:
            return [], [], 0.0

        max_chaos_score = max(score for _, score, _ in chaos_scores)
        norm_max_chaos = min(1.0, max_chaos_score / self.CHAOS_NORMALIZER)

        # Vertical-flow timeline is always returned and is independent of the
        # chaos floor — mirroring (calm downward camera tilt) does not require
        # chaotic motion to be a true positive.
        vertical_flow_timeline = [(t, v) for t, _, v in chaos_scores]

        if max_chaos_score < self.CHAOS_FLOOR:
            return [], vertical_flow_timeline, float(norm_max_chaos)

        threshold = max_chaos_score * self.SPIKE_RELATIVE_THRESHOLD
        spikes = [{'time': ts, 'v_flow': v}
                  for ts, score, v in chaos_scores if score > threshold]

        return spikes, vertical_flow_timeline, float(norm_max_chaos)

    def _select_pose_detection(self, results, crop_w, crop_h):
        """Pick the pose detection whose bbox most aligns with the input crop
        rectangle (highest IoU with (0,0,crop_w,crop_h)). Returns the keypoints
        tensor for the selected person, or None if no usable detection exists.
        """
        if not results:
            return None
        r = results[0]
        if not getattr(r, 'keypoints', None):
            return None
        # Resolved Issue 15: the no-detection guard formerly checked
        # shape[1] (always 17), so zero-detection frames raised IndexError
        # downstream. Check person count via shape[0].
        if r.keypoints.data is None or r.keypoints.data.shape[0] == 0:
            return None
        if r.keypoints.data.shape[1] < 17:
            return None

        n = r.keypoints.data.shape[0]
        if n == 1:
            return r.keypoints.data[0].cpu().numpy()

        # IoU vs full-crop reference. If boxes are missing, fall back to
        # largest-area in keypoint span (Resolved Issue 13).
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and getattr(boxes, 'xyxy', None) is not None and len(boxes.xyxy) == n:
            ref = (0, 0, crop_w, crop_h)
            best_idx = 0
            best_iou = -1.0
            for i in range(n):
                bx = boxes.xyxy[i].cpu().numpy().tolist()
                iou = self._iou(bx, ref)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            return r.keypoints.data[best_idx].cpu().numpy()

        # Fallback: use keypoint spatial spread as a proxy for size.
        best_idx = 0
        best_area = -1.0
        for i in range(n):
            kp = r.keypoints.data[i].cpu().numpy()
            xs = kp[:, 0][kp[:, 2] > 0]
            ys = kp[:, 1][kp[:, 2] > 0]
            if len(xs) == 0:
                continue
            area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            if area > best_area:
                best_area = area
                best_idx = i
        return r.keypoints.data[best_idx].cpu().numpy()

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return float(inter / union)

    def _extract_and_correlate_pose(self, video_path, timestamps, bboxes,
                                    start_sec, end_sec, ego_spikes,
                                    vertical_flow_timeline):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                return None

            valid_frames = []
            for t, bbox in zip(timestamps, bboxes):
                if start_sec <= t <= end_sec:
                    valid_frames.append((t, bbox))

            valid_frames.sort(key=lambda x: x[0])

            pose_velocities = []
            pose_spine_angles = []
            prev_kpts_by_idx = None
            prev_t = None

            for t, bbox in valid_frames:
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop_w = x2 - x1
                crop_h = y2 - y1
                crop_diag = float(np.sqrt(crop_w ** 2 + crop_h ** 2))
                if crop_diag == 0:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                results = self.model(crop, device=self.device, verbose=False)
                kpts = self._select_pose_detection(results, crop_w, crop_h)
                if kpts is None:
                    continue

                current_kpts_by_idx = {}
                for k in self.RELEVANT_KPT_INDICES:
                    if k < len(kpts):
                        x, y, conf = kpts[k]
                        if conf > self.KPT_CONFIDENCE_THRESHOLD:
                            current_kpts_by_idx[k] = (x / crop_diag, y / crop_diag)

                if current_kpts_by_idx:
                    shoulders_x = [current_kpts_by_idx[k][0] for k in (5, 6) if k in current_kpts_by_idx]
                    shoulders_y = [current_kpts_by_idx[k][1] for k in (5, 6) if k in current_kpts_by_idx]
                    hips_x = [current_kpts_by_idx[k][0] for k in (11, 12) if k in current_kpts_by_idx]
                    hips_y = [current_kpts_by_idx[k][1] for k in (11, 12) if k in current_kpts_by_idx]

                    if shoulders_x and hips_x:
                        sx, sy = np.mean(shoulders_x), np.mean(shoulders_y)
                        hx, hy = np.mean(hips_x), np.mean(hips_y)
                        angle = np.arctan2(sy - hy, sx - hx)
                        pose_spine_angles.append((t, angle))

                if current_kpts_by_idx and prev_kpts_by_idx is not None and prev_t is not None:
                    dt = t - prev_t
                    if 0 < dt <= self.PREV_KPTS_CARRY_FORWARD_SEC:
                        common_keys = set(current_kpts_by_idx.keys()) & set(prev_kpts_by_idx.keys())
                        distances = []
                        for k in common_keys:
                            cx, cy = current_kpts_by_idx[k]
                            px, py = prev_kpts_by_idx[k]
                            distances.append(float(np.sqrt((cx - px) ** 2 + (cy - py) ** 2)))
                        if distances:
                            pose_velocities.append((t, float(np.mean(distances) / dt)))

                # Resolved Issue 17: only advance prev when the current frame
                # produced usable keypoints. A failed-detection frame leaves
                # prev_* untouched so the next successful frame still has a
                # reference point; the per-step dt cap above prevents stale
                # comparisons across long occlusions.
                if current_kpts_by_idx:
                    prev_kpts_by_idx = current_kpts_by_idx
                    prev_t = t
        finally:
            cap.release()

        if not pose_velocities and not pose_spine_angles:
            return None

        if pose_velocities:
            max_vel = 0.0
            peak_t = None
            for t, vel in pose_velocities:
                if vel > max_vel:
                    max_vel = vel
                    peak_t = t

            norm_peak_vel = float(min(self.VELOCITY_CAP, max_vel / self.VELOCITY_NORMALIZER))
        else:
            max_vel = 0.0
            peak_t = None
            norm_peak_vel = 0.0

        resonance_detected = False
        min_delay = float('inf')

        if peak_t is not None:
            for spike in ego_spikes:
                spike_t = spike['time'] if isinstance(spike, dict) else spike
                delay = peak_t - spike_t
                if 0 < delay <= self.RESONANCE_WINDOW_SEC:
                    resonance_detected = True
                    if delay < min_delay:
                        min_delay = delay

        if not resonance_detected:
            min_delay = 0.0

        empathy_scalar = 0.0
        if resonance_detected:
            empathy_scalar = float(min(1.0, norm_peak_vel / self.EMPATHY_NORMALIZER))

        mirroring_detected, mirroring_scalar = self._correlate_mirroring(
            vertical_flow_timeline, pose_spine_angles,
        )

        return {
            'velocity_peak': round(norm_peak_vel, 2),
            'delay_sec': round(min_delay, 2),
            'resonance_detected': resonance_detected,
            'empathy_scalar': round(empathy_scalar, 2),
            'mirroring_detected': mirroring_detected,
            'mirroring_scalar': round(mirroring_scalar, 2),
        }

    def _correlate_mirroring(self, vertical_flow_timeline, pose_spine_angles):
        """Scan the vertical-flow timeline (independent of the chaos floor) for
        sustained downward camera motion and check whether the bystander's
        spine angle changed congruently within ±MIRRORING_TIME_WINDOW_SEC.
        """
        mirroring_detected = False
        mirroring_scalar = 0.0

        if not vertical_flow_timeline or not pose_spine_angles:
            return mirroring_detected, mirroring_scalar

        for t_v, v_flow in vertical_flow_timeline:
            if v_flow >= self.MIRRORING_VFLOW_THRESHOLD:
                continue

            window = self.MIRRORING_TIME_WINDOW_SEC
            angles_before = [a for t, a in pose_spine_angles if t_v - window <= t <= t_v]
            angles_after = [a for t, a in pose_spine_angles if t_v < t <= t_v + window]

            if not angles_before or not angles_after:
                continue

            mean_before = float(np.mean(angles_before))
            mean_after = float(np.mean(angles_after))
            delta = abs(mean_after - mean_before)

            if delta > self.MIRRORING_DELTA_THRESHOLD:
                mirroring_detected = True
                mirroring_scalar = float(min(1.0, delta / self.MIRRORING_SCALAR_NORMALIZER))
                break

        return mirroring_detected, mirroring_scalar
