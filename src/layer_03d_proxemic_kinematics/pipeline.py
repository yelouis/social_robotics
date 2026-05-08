import os
from pathlib import Path

# Set HuggingFace Cache to Extreme SSD to prevent local drive fillup
# This must be set BEFORE transformers is imported
SSD_HF_CACHE = "/Volumes/Extreme SSD/huggingface_cache"
os.environ['HF_HOME'] = SSD_HF_CACHE

import json
import cv2
import traceback
import numpy as np
from PIL import Image

try:
    import torch
    from transformers import pipeline, SamModel, SamProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ProxemicKinematicsPipeline:
    # --- Tuning Constants ---
    # All proxemic heuristic thresholds live here so retuning is a single-file
    # surgical edit (or a subclass override) rather than a hunt across helpers.
    OPTICAL_FLOW_NOISE_THRESHOLD = 15.0   # 95th percentile farneback magnitude beyond which we void the proxemic vector
    BBOX_NORM_PCT = 50.0                  # bbox area delta % that maps to ±1.0 after normalization
    DEPTH_NORM_SCALE = 2.0                # multiplier on -depth_delta when normalizing into [-1, 1]
    BBOX_WEIGHT = 0.4                     # weight of bbox heuristic in the fused proxemic vector
    DEPTH_WEIGHT = 0.6                    # weight of depth heuristic in the fused proxemic vector
    MICROMOVEMENT_THRESHOLD = 0.05        # |vector| below this is treated as no movement (jitter rejection)
    APPROACH_THRESHOLD = 0.3              # vector > this -> Approach_Intervention; < -this -> Avoidance

    def __init__(self, input_manifest_path, output_result_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03d_proxemic_kinematics_errors.json"
        self.force = force
        self.depth_estimator = None
        self.sam_model = None
        self.sam_processor = None
        self.device = 'cpu'
        
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
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Missing required dependency 'transformers>=4.35.0'. "
                               "Install with: pip install transformers>=4.35.0 huggingface_hub torch")

        # Validate the Extreme SSD is actually mounted before letting transformers
        # write multi-hundred-MB model weights into a phantom /Volumes/<name>
        # directory on the internal drive. macOS does not block writes under
        # /Volumes/<name> when nothing is mounted there, so a missing SSD silently
        # fills the boot disk — exactly the failure mode the SSD cache exists to
        # prevent.
        ssd_root = SSD_HF_CACHE.rsplit("/huggingface_cache", 1)[0]
        if not os.path.ismount(ssd_root):
            raise RuntimeError(
                f"Extreme SSD is not mounted at '{ssd_root}'. The Proxemic Kinematics "
                f"layer caches Depth Anything V2 + SAM weights (~500MB) on this volume "
                f"to keep the boot disk clear. Mount the SSD and retry, or override "
                f"SSD_HF_CACHE in pipeline.py if running on a host without the SSD."
            )

        try:
            # Create SSD cache directory if it doesn't exist
            os.makedirs(SSD_HF_CACHE, exist_ok=True)

            # Check for MPS/GPU
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'

            device_id = -1
            if self.device == 'mps':
                device_id = 'mps'
            elif self.device == 'cuda':
                device_id = 0

            print(f"Initializing Depth Anything V2-Small on {self.device} (Cache: {SSD_HF_CACHE})...")
            self.depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device_id)

            print(f"Initializing SAM-1 (bbox-prompted) on {self.device}...")
            # Bbox-prompted SAM via the lower-level SamModel/SamProcessor API.
            # The high-level mask-generation pipeline runs an exhaustive 32x32
            # candidate-point grid (1024 forward passes per crop), which dominates
            # wall-clock cost on the M4 Pro MPS backend and is wasted work given
            # we already have the bystander bbox as a single-mask prompt.
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
            self.sam_model.eval()
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

            # Validate download path
            model_cache_path = Path(SSD_HF_CACHE) / "hub" / "models--depth-anything--Depth-Anything-V2-Small-hf"
            if not model_cache_path.exists():
                print("Warning: Model does not appear to be saved in the expected Extreme SSD cache location.")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Depth Anything V2: {e}")

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
            except:
                pass
        
        errors.append(error_entry)
        
        temp_err = self.error_log_path.with_suffix('.tmp')
        with open(temp_err, 'w') as f:
            json.dump(errors, f, indent=4)
        temp_err.replace(self.error_log_path)

    def run(self):
        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)

        results = []
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    results = json.load(f)
            except:
                pass

        for entry in registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue

            print(f"Processing Proxemic Kinematics for video: {video_id}")
            try:
                result = self.process_video(entry)
                if result is None:
                    # Sentinel record: persist the skip decision so subsequent
                    # resume runs don't redo the (expensive) optical-flow + depth
                    # scans only to discard the result again. Downstream consumers
                    # filter on `tasks_analyzed` length or `skipped_reason`.
                    result = {
                        "video_id": video_id,
                        "layer": "03d_proxemic_kinematics",
                        "tasks_analyzed": [],
                        "skipped_reason": "no_output_produced"
                    }
                results.append(result)
                self.processed_ids.add(video_id)

                # Atomic write
                temp_out = self.output_result_path.with_suffix('.tmp')
                with open(temp_out, 'w') as f:
                    json.dump(results, f, indent=4)
                temp_out.replace(self.output_result_path)
            except Exception as e:
                self._log_error(video_id, e)
            finally:
                # Bound MPS/CUDA cache growth at one-video granularity. Long
                # batches accumulate intermediate activations across Depth
                # Anything + SAM forward passes; flushing here keeps the 24GB
                # M4 Pro shared budget from sliding into thermal throttle.
                self._release_accelerator_cache()

        print(f"Final count: {len(results)} videos processed for Proxemic Kinematics.")

    def _release_accelerator_cache(self):
        """Flush the per-device cache after each video. No-op on CPU."""
        if not TRANSFORMERS_AVAILABLE:
            return
        try:
            if self.device == 'mps' and hasattr(torch, 'mps'):
                torch.mps.empty_cache()
            elif self.device == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            # Cache flush is best-effort; never let it break the run loop.
            pass

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['video_path'])
        
        if not video_path.exists():
            print(f"File not found: {video_path}")
            return None
            
        bystanders = entry.get('bystander_detections', [])
        tasks = entry.get('identified_tasks', [])
        
        if not bystanders or not tasks:
            print(f"No bystanders or tasks found for {video_id}.")
            return None
            
        tasks_analyzed = []
        for task in tasks:
            task_id = task.get('task_id', 'unknown')
            meta = task.get('task_temporal_metadata', {})
            reaction_window = meta.get('task_reaction_window_sec')
            
            if not reaction_window or len(reaction_window) != 2:
                continue
                
            start_sec, end_sec = reaction_window
            
            chaos_score = self._extract_ego_motion_noise(video_path, start_sec, end_sec)
            noise_threshold = self.OPTICAL_FLOW_NOISE_THRESHOLD

            per_person = []
            for bystander in bystanders:
                person_id = bystander.get('person_id')
                timestamps_sec = bystander.get('timestamps_sec', [])
                bounding_boxes = bystander.get('bounding_boxes', [])
                
                if not timestamps_sec or not bounding_boxes:
                    continue
                    
                bbox_delta = self._calculate_bbox_scale_delta(timestamps_sec, bounding_boxes, start_sec, end_sec)
                
                # Check if person is present in the window at all
                if bbox_delta is None:
                    continue
                    
                if chaos_score > noise_threshold:
                    per_person.append({
                        "person_id": person_id,
                        "bbox_scale_delta_pct": round(bbox_delta, 2),
                        "depth_anything_v2_delta": 0.0,
                        "proxemic_vector": 0.0,
                        "classified_action": "Neutral",
                        "proxemic_confidence": 0.0,
                        "optical_flow_noise": round(chaos_score, 2)
                    })
                    continue
                    
                depth_delta = self._calculate_depth_delta(video_path, timestamps_sec, bounding_boxes, start_sec, end_sec)
                
                # Check if depth calculation succeeded
                if depth_delta is None:
                    continue
                    
                proxemic_vector, action = self._compute_proxemic_vector(bbox_delta, depth_delta)
                
                # Confidence score: if signs disagree, lower confidence
                # e.g., bbox grows (+) but depth increases (-) -> conflict
                # bbox_delta: + approach. depth_delta: - approach.
                # signs agree if (bbox_delta > 0 and depth_delta < 0) or (bbox_delta < 0 and depth_delta > 0)
                # sign of bbox_delta vs sign of (-depth_delta)
                bbox_sign = 1 if bbox_delta > 0 else (-1 if bbox_delta < 0 else 0)
                depth_app_sign = 1 if depth_delta < 0 else (-1 if depth_delta > 0 else 0)
                
                confidence = 1.0 - (abs(bbox_sign - depth_app_sign) / 2.0)
                
                per_person.append({
                    "person_id": person_id,
                    "bbox_scale_delta_pct": round(bbox_delta, 2),
                    "depth_anything_v2_delta": round(depth_delta, 4),
                    "proxemic_vector": round(proxemic_vector, 2),
                    "classified_action": action,
                    "proxemic_confidence": round(confidence, 2),
                    "optical_flow_noise": round(chaos_score, 2)
                })
                
            if per_person:
                tasks_analyzed.append({
                    "task_id": task_id,
                    "per_person": per_person
                })
                
        if not tasks_analyzed:
            return None
            
        return {
            "video_id": video_id,
            "layer": "03d_proxemic_kinematics",
            "tasks_analyzed": tasks_analyzed
        }

    def _calculate_bbox_scale_delta(self, timestamps, bboxes, start_sec, end_sec):
        # Extract bboxes in the window
        window_areas = []
        for t, bbox in zip(timestamps, bboxes):
            if start_sec <= t <= end_sec:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                window_areas.append((t, area))
                
        if len(window_areas) < 2:
            return None
            
        window_areas.sort(key=lambda x: x[0])
        first_area = window_areas[0][1]
        last_area = window_areas[-1][1]
        
        if first_area <= 0:
            return 0.0
            
        delta_pct = ((last_area - first_area) / first_area) * 100.0
        return delta_pct

    def _extract_ego_motion_noise(self, video_path, start_sec, end_sec):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            return 0.0
            
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return 0.0
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
        
        max_chaos = 0.0
        current_frame_idx = start_frame + 1
        
        while current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            chaos_score = float(np.percentile(mag, 95))
            if chaos_score > max_chaos:
                max_chaos = chaos_score
                
            prev_gray = gray
            current_frame_idx += 1
            
        cap.release()
        return max_chaos

    def _calculate_depth_delta(self, video_path, timestamps, bboxes, start_sec, end_sec):
        if self.depth_estimator is None:
            return None
            
        # Select timestamps in window
        valid_frames = []
        for t, bbox in zip(timestamps, bboxes):
            if start_sec <= t <= end_sec:
                valid_frames.append((t, bbox))
                
        if len(valid_frames) < 2:
            return None
            
        valid_frames.sort(key=lambda x: x[0])
        
        window_duration = end_sec - start_sec
        num_frames = max(5, int(window_duration * 3))
        num_frames = min(20, num_frames)
        
        if len(valid_frames) > num_frames:
            indices = np.linspace(0, len(valid_frames) - 1, num_frames, dtype=int)
            valid_frames = [valid_frames[i] for i in indices]
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            return None
            
        depths = []
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
                
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                # depth result is dict with 'depth' key containing PIL image
                result = self.depth_estimator(img)
                depth_img = result['depth']
                depth_arr = np.array(depth_img)
                
                # The depth map may be a different resolution than the
                # original frame.  Rescale bbox coordinates to match.
                dh, dw = depth_arr.shape[:2]
                scale_x = dw / w
                scale_y = dh / h
                dx1 = max(0, int(x1 * scale_x))
                dy1 = max(0, int(y1 * scale_y))
                dx2 = min(dw, int(x2 * scale_x))
                dy2 = min(dh, int(y2 * scale_y))
                
                # SAM-1 bbox-prompted segmentation. We pass the bystander bbox in
                # original frame coordinates as an `input_boxes` prompt and take
                # the highest-IoU mask from the (1, num_masks_per_box, H, W)
                # output. One forward pass per frame instead of 1024 candidate
                # points; deterministic mask selection by IoU, not by largest-
                # area heuristic gambling on background.
                best_mask_full = self._segment_with_sam(img, x1, y1, x2, y2)
                if best_mask_full is not None:
                    # The SAM mask is at the original frame resolution; resize
                    # to the depth map's resolution so masking aligns 1:1.
                    mask_pil = Image.fromarray(best_mask_full.astype(np.uint8) * 255).resize(
                        (dw, dh), Image.NEAREST
                    )
                    mask = np.array(mask_pil, dtype=bool)
                    # If SAM returned an empty mask (degenerate prompt), fall
                    # back to the rectangular bbox so we still produce a sample.
                    if not mask.any():
                        mask = np.zeros(depth_arr.shape[:2], dtype=bool)
                        mask[dy1:dy2, dx1:dx2] = True
                else:
                    # SAM unavailable or inference failed — fall back to the bbox.
                    mask = np.zeros(depth_arr.shape[:2], dtype=bool)
                    mask[dy1:dy2, dx1:dx2] = True

                masked_depths = depth_arr[mask]
                if masked_depths.size > 0:
                    median_depth = float(np.median(masked_depths))
                    # normalize by 255 for simplicity
                    depths.append((t, median_depth / 255.0))
            except Exception as e:
                print(f"Depth inference failed at t={t}: {e}")
                
        cap.release()

        return self._slope_span_delta(depths)

    @staticmethod
    def _slope_span_delta(depths):
        """Linear-regression slope through (t, median_depth) pairs, scaled by
        window duration to recover a unit-consistent "predicted span" delta.

        Robust to single-frame outliers at either endpoint and uses all
        collected samples — recovering the rationale for the adaptive 3 FPS
        sampling that previously fed a discarded two-point delta.
        """
        if len(depths) < 2:
            return None
        ts = np.array([t for t, _ in depths], dtype=float)
        ds = np.array([d for _, d in depths], dtype=float)
        slope, _ = np.polyfit(ts, ds, 1)
        window_duration = ts[-1] - ts[0]
        if window_duration <= 0:
            return None
        return float(slope * window_duration)

    def _segment_with_sam(self, img, x1, y1, x2, y2):
        """Run SAM with the bbox as an input-box prompt; return a boolean mask
        at the original image resolution, or None if SAM is unavailable / fails.

        The mask is selected by the SAM IoU score head rather than by largest
        area, so we don't accidentally promote a background mask just because
        it covers more pixels.
        """
        if self.sam_model is None or self.sam_processor is None:
            return None
        try:
            inputs = self.sam_processor(
                img,
                input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.sam_model(**inputs)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            # masks[0] shape: (num_prompts=1, num_masks_per_prompt, H, W)
            mask_tensor = masks[0][0]
            iou_scores = outputs.iou_scores.cpu().numpy().reshape(-1)
            best_idx = int(np.argmax(iou_scores))
            return mask_tensor[best_idx].numpy().astype(bool)
        except Exception as e:
            print(f"SAM bbox-prompt inference failed: {e}")
            return None

    def _compute_proxemic_vector(self, bbox_delta, depth_delta):
        # Heuristic combination of bbox-area % delta and depth slope-span.
        # Decreasing depth (depth_delta < 0) -> approach -> positive vector.
        norm_bbox = max(-1.0, min(1.0, bbox_delta / self.BBOX_NORM_PCT))
        norm_depth = max(-1.0, min(1.0, -depth_delta * self.DEPTH_NORM_SCALE))
        vector = (norm_bbox * self.BBOX_WEIGHT) + (norm_depth * self.DEPTH_WEIGHT)

        # Reject jitter inside the deadband.
        if abs(vector) < self.MICROMOVEMENT_THRESHOLD:
            vector = 0.0

        if vector > self.APPROACH_THRESHOLD:
            action = "Approach_Intervention"
        elif vector < -self.APPROACH_THRESHOLD:
            action = "Avoidance"
        else:
            action = "Neutral"

        return vector, action
