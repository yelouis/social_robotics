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
            # Assuming yolov8n-pose.pt is in the root directory
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_path = base_dir / "yolov8n-pose.pt"
            
            if not model_path.exists():
                print(f"Warning: Model {model_path} not found. Attempting to download via ultralytics...")
                
            self.model = YOLO(str(model_path) if model_path.exists() else "yolov8n-pose.pt")
            # We don't strictly need to set the device here as ultralytics auto-selects, 
            # but we can pass it during inference.
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
                if result:
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
            
            # Step 1: EgoMotion Extraction
            ego_spikes, max_chaos_score = self._extract_ego_motion(video_path, start_sec, end_sec)
            
            if not ego_spikes:
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
                    video_path, timestamps_sec, bounding_boxes, start_sec, end_sec, ego_spikes
                )
                
                if pose_analysis:
                    per_person.append({
                        "person_id": person_id,
                        "bystander_pose_velocity_peak": pose_analysis['velocity_peak'],
                        "resonance_delay_sec": pose_analysis['delay_sec'],
                        "motor_resonance_detected": pose_analysis['resonance_detected'],
                        "empathy_scalar": pose_analysis['empathy_scalar']
                    })
                
            if per_person:
                tasks_analyzed.append({
                    "task_id": task_id,
                    "ego_kinetic_chaos_score": round(max_chaos_score, 2),
                    "per_person": per_person
                })
                
        if not tasks_analyzed:
            return None
            
        return {
            "video_id": video_id,
            "layer": "03f_motor_resonance",
            "tasks_analyzed": tasks_analyzed
        }

    def _extract_ego_motion(self, video_path, start_sec, end_sec):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], 0.0
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            return [], 0.0
            
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return [], 0.0
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5) # Downsample for speed
        
        chaos_scores = []
        current_frame_idx = start_frame + 1
        
        while current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # The "chaos" score can be the standard deviation or high percentile of the magnitude
            chaos_score = np.percentile(mag, 95)
            timestamp = current_frame_idx / fps
            chaos_scores.append((timestamp, chaos_score))
            
            prev_gray = gray
            current_frame_idx += 1
            
        cap.release()
        
        if not chaos_scores:
            return [], 0.0
            
        # Find spikes in chaos score
        max_chaos_score = max([score for _, score in chaos_scores])
        
        # Normalize max chaos score (assume 20 pixels/frame is very high optical flow)
        norm_max_chaos = min(1.0, max_chaos_score / 20.0)
        
        # If the max chaos is below a meaningful threshold, there is no real
        # camera jolt—return empty spikes to avoid false-positive resonance.
        if max_chaos_score < 3.0:
            return [], float(norm_max_chaos)
        
        # A spike is anything > 70% of the max chaos score in this window
        spikes = [ts for ts, score in chaos_scores if score > max_chaos_score * 0.7]
        
        return spikes, float(norm_max_chaos)

    def _extract_and_correlate_pose(self, video_path, timestamps, bboxes, start_sec, end_sec, ego_spikes):
        # We need to run YOLO pose on the bounding box crops
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            return None
        
        valid_frames = []
        for t, bbox in zip(timestamps, bboxes):
            if start_sec <= t <= end_sec:
                valid_frames.append((t, bbox))
                
        valid_frames.sort(key=lambda x: x[0])
        
        pose_velocities = []
        # Use a dict keyed by keypoint index to guarantee cross-frame alignment.
        # Previously, only high-confidence keypoints were appended to a flat list,
        # so index 0 could be "left shoulder" in one frame and "right wrist" in the
        # next—producing meaningless velocity values between different body parts.
        prev_kpts_by_idx = None  # dict: kpt_index -> (x, y)
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
            crop_diag = np.sqrt(crop_w**2 + crop_h**2)
            if crop_diag == 0:
                continue
                
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # Run YOLO pose on crop
            results = self.model(crop, device=self.device, verbose=False)
            
            if not results or not results[0].keypoints or results[0].keypoints.data.shape[1] == 0:
                continue
                
            # Get keypoints (shape: N, 17, 3 -> x, y, conf)
            # YOLOv8 pose keypoints: 5=L shoulder, 6=R shoulder, 9=L wrist, 10=R wrist
            kpts = results[0].keypoints.data[0].cpu().numpy() # take first person in crop
            
            relevant_kpts = [5, 6, 9, 10]
            current_kpts_by_idx = {}
            for k in relevant_kpts:
                if k < len(kpts):
                    x, y, conf = kpts[k]
                    if conf > 0.5:
                        # Normalize by crop diagonal so velocity is scale-invariant
                        current_kpts_by_idx[k] = (x / crop_diag, y / crop_diag)
            
            if current_kpts_by_idx and prev_kpts_by_idx is not None and prev_t is not None:
                # Calculate average velocity only for keypoints present in BOTH frames
                dt = t - prev_t
                if dt > 0:
                    distances = []
                    common_keys = set(current_kpts_by_idx.keys()) & set(prev_kpts_by_idx.keys())
                    for k in common_keys:
                        cx, cy = current_kpts_by_idx[k]
                        px, py = prev_kpts_by_idx[k]
                        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                        distances.append(dist)
                        
                    if distances:
                        avg_dist = np.mean(distances)
                        velocity = avg_dist / dt
                        pose_velocities.append((t, velocity))
                        
            prev_kpts_by_idx = current_kpts_by_idx
            prev_t = t
            
        cap.release()
        
        if not pose_velocities:
            return None
            
        # Find peak velocity
        max_vel = 0.0
        peak_t = None
        for t, vel in pose_velocities:
            if vel > max_vel:
                max_vel = vel
                peak_t = t
                
        # Normalize velocity. Since keypoints are now divided by crop diagonal,
        # max_vel is in "diagonals per second".  A value of ~1.0 means the
        # keypoints moved one full crop-diagonal in one second—very fast.
        norm_peak_vel = float(min(10.0, max_vel / 0.5))
        
        # Correlate with ego spikes
        resonance_detected = False
        min_delay = float('inf')
        
        if peak_t is not None:
            for spike_t in ego_spikes:
                delay = peak_t - spike_t
                if 0 < delay <= 0.5: # 0.5s reaction window
                    resonance_detected = True
                    if delay < min_delay:
                        min_delay = delay
                        
        if not resonance_detected:
            min_delay = 0.0
            
        empathy_scalar = 0.0
        if resonance_detected:
            empathy_scalar = float(min(1.0, norm_peak_vel / 5.0))
            
        return {
            'velocity_peak': round(norm_peak_vel, 2),
            'delay_sec': round(min_delay, 2),
            'resonance_detected': resonance_detected,
            'empathy_scalar': round(empathy_scalar, 2)
        }
