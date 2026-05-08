import json
import traceback
import cv2
import numpy as np
from pathlib import Path


class SharedRealityPipeline:
    # --- Detection Tuning ---
    OPTICAL_FLOW_DOWNSAMPLE = 0.5
    TARGET_FLOW_FPS = 10.0
    CENTERING_LOWER_BOUND = 0.35
    CENTERING_UPPER_BOUND = 0.65
    SHIFT_THRESHOLD_RATIO = 0.04
    FINAL_CENTERING_TAIL_FRACTION = 0.25

    @property
    def OPTICAL_FLOW_UPSCALE(self):
        # Derived to enforce the inverse coupling with OPTICAL_FLOW_DOWNSAMPLE.
        return 1.0 / self.OPTICAL_FLOW_DOWNSAMPLE

    def __init__(self, input_manifest_path, output_result_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03g_shared_reality_errors.json"
        self.force = force
        self.processed_ids = set()

        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting fresh.")

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
            "layer": "03g_shared_reality",
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

            print(f"Processing Shared Reality for video: {video_id}")
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

        print(f"Final count: {len(results)} videos processed for Shared Reality.")

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['video_path'])

        if not video_path.exists():
            print(f"File not found: {video_path}")
            return self._sentinel(video_id, "missing_video")

        bystanders = entry.get('bystander_detections', [])
        tasks = entry.get('identified_tasks', [])

        if not bystanders or not tasks:
            print(f"No bystanders or tasks found for {video_id}.")
            return self._sentinel(video_id, "no_bystanders_or_tasks")

        # Open the capture once to read frame metadata, then close. Both
        # helpers consume these values directly so neither needs to re-open
        # the file (Resolved Issue 13).
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self._sentinel(video_id, "missing_video")
        try:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()

        if width == 0 or height == 0 or fps == 0:
            return self._sentinel(video_id, "missing_video")

        frame_diagonal = float(np.sqrt(width ** 2 + height ** 2))
        threshold = frame_diagonal * self.SHIFT_THRESHOLD_RATIO

        tasks_analyzed = []
        for task in tasks:
            task_id = task.get('task_id', 'unknown')
            meta = task.get('task_temporal_metadata', {})
            reaction_window = meta.get('task_reaction_window_sec')

            if not reaction_window or len(reaction_window) != 2:
                continue

            start_sec, end_sec = reaction_window

            shift_vector = self._extract_camera_shift(
                video_path, start_sec, end_sec, width, height, fps, bystanders,
            )
            bystander_centered = self._check_bystander_centering(
                bystanders, start_sec, end_sec, width, height,
            )

            shift_magnitude = float(np.sqrt(shift_vector[0] ** 2 + shift_vector[1] ** 2))
            social_reference_sought = bool(bystander_centered and shift_magnitude > threshold)

            tasks_analyzed.append({
                "task_id": task_id,
                "post_climax_camera_shift_vector": shift_vector,
                "bystander_centered_in_fov": bystander_centered,
                "social_reference_sought": social_reference_sought,
            })

        if not tasks_analyzed:
            return self._sentinel(video_id, "no_valid_tasks")

        return {
            "video_id": video_id,
            "layer": "03g_shared_reality",
            "tasks_analyzed": tasks_analyzed,
        }

    def _bystander_mask_for_frame(self, t, frame_shape_ds, bystanders):
        """Boolean mask zeroing out the nearest-timestamp bbox for every
        bystander on the *downsampled* frame. ``True`` = keep (background)."""
        h_ds, w_ds = frame_shape_ds
        keep = np.ones((h_ds, w_ds), dtype=bool)
        scale = self.OPTICAL_FLOW_DOWNSAMPLE
        for b in bystanders:
            ts = b.get('timestamps_sec', [])
            bxs = b.get('bounding_boxes', [])
            if not ts or not bxs:
                continue
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

    def _extract_camera_shift(self, video_path, start_sec, end_sec,
                              width, height, fps, bystanders):
        """Estimate accumulated camera shift via Farneback optical flow over
        background pixels (Resolved Issues 14, 15)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [0.0, 0.0]
        try:
            if fps <= 0:
                return [0.0, 0.0]

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            frame_stride = max(1, int(round(fps / self.TARGET_FLOW_FPS)))
            upscale = self.OPTICAL_FLOW_UPSCALE

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, prev_frame = cap.read()
            if not ret:
                return [0.0, 0.0]

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.resize(
                prev_gray, (0, 0),
                fx=self.OPTICAL_FLOW_DOWNSAMPLE,
                fy=self.OPTICAL_FLOW_DOWNSAMPLE,
            )

            current_frame_idx = start_frame + frame_stride

            total_dx = 0.0
            total_dy = 0.0

            while current_frame_idx <= end_frame:
                if frame_stride > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(
                    gray, (0, 0),
                    fx=self.OPTICAL_FLOW_DOWNSAMPLE,
                    fy=self.OPTICAL_FLOW_DOWNSAMPLE,
                )

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                )

                # Mask out bystander regions so the mean reflects only
                # background motion (Resolved Issue 14).
                t = current_frame_idx / fps
                mask = self._bystander_mask_for_frame(t, gray.shape, bystanders)
                if mask.any():
                    mean_dx = -float(np.mean(flow[..., 0][mask]))
                    mean_dy = -float(np.mean(flow[..., 1][mask]))
                else:
                    mean_dx = -float(np.mean(flow[..., 0]))
                    mean_dy = -float(np.mean(flow[..., 1]))

                # The flow between stride-spaced frames already represents the
                # full inter-frame displacement, so accumulating per iteration
                # without re-multiplying by stride yields the same total as
                # a per-frame loop for continuous motion.
                total_dx += mean_dx * upscale
                total_dy += mean_dy * upscale

                prev_gray = gray
                current_frame_idx += frame_stride
        finally:
            cap.release()

        return [round(total_dx, 2), round(total_dy, 2)]

    def _check_bystander_centering(self, bystanders, start_sec, end_sec, width, height):
        """Return True iff a bystander centroid lands in the central
        [CENTERING_LOWER_BOUND, CENTERING_UPPER_BOUND] region during the
        *final FINAL_CENTERING_TAIL_FRACTION of the reaction window*. The
        tail constraint encodes the documented "pan away then center"
        semantic and rules out pre-centered-bystander false positives
        (Resolved Issue 11)."""
        if not bystanders:
            return False
        if width <= 0 or height <= 0:
            return False

        window_duration = end_sec - start_sec
        if window_duration <= 0:
            return False
        tail_start = end_sec - window_duration * self.FINAL_CENTERING_TAIL_FRACTION

        min_x = width * self.CENTERING_LOWER_BOUND
        max_x = width * self.CENTERING_UPPER_BOUND
        min_y = height * self.CENTERING_LOWER_BOUND
        max_y = height * self.CENTERING_UPPER_BOUND

        for bystander in bystanders:
            timestamps = bystander.get('timestamps_sec', [])
            bboxes = bystander.get('bounding_boxes', [])

            if len(timestamps) != len(bboxes):
                continue

            for t, bbox in zip(timestamps, bboxes):
                if tail_start <= t <= end_sec:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    if min_x <= cx <= max_x and min_y <= cy <= max_y:
                        return True

        return False
