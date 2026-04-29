import os
import json
import cv2
import traceback
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ProxemicKinematicsPipeline:
    def __init__(self, input_manifest_path, output_result_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03d_proxemic_kinematics_errors.json"
        self.force = force
        self.depth_estimator = None
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
            raise RuntimeError("transformers and torch are required for the Proxemic Kinematics layer. "
                               "Please install them via 'pip install transformers torch torchvision huggingface_hub'.")

        try:
            # Set HuggingFace Cache to Extreme SSD to prevent local drive fillup
            ssd_hf_cache = "/Volumes/Extreme SSD/huggingface_cache"
            os.makedirs(ssd_hf_cache, exist_ok=True)
            os.environ['HF_HOME'] = ssd_hf_cache

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
                
            print(f"Initializing Depth Anything V2-Small on {self.device} (Cache: {ssd_hf_cache})...")
            self.depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device_id)
            
            # Validate download path
            model_cache_path = Path(ssd_hf_cache) / "hub" / "models--depth-anything--Depth-Anything-V2-Small-hf"
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

        print(f"Final count: {len(results)} videos processed for Proxemic Kinematics.")

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
                    
                depth_delta = self._calculate_depth_delta(video_path, timestamps_sec, bounding_boxes, start_sec, end_sec)
                
                # Check if depth calculation succeeded
                if depth_delta is None:
                    continue
                    
                proxemic_vector, action = self._compute_proxemic_vector(bbox_delta, depth_delta)
                
                per_person.append({
                    "person_id": person_id,
                    "bbox_scale_delta_pct": round(bbox_delta, 2),
                    "depth_anything_v2_delta": round(depth_delta, 4),
                    "proxemic_vector": round(proxemic_vector, 2),
                    "classified_action": action
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
        
        # Take up to 5 frames spread out to save compute
        if len(valid_frames) > 5:
            indices = np.linspace(0, len(valid_frames) - 1, 5, dtype=int)
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
                
                # Mask with rescaled bbox
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
        
        if len(depths) < 2:
            return None
            
        # delta = last_depth - first_depth
        first_depth = depths[0][1]
        last_depth = depths[-1][1]
        
        return last_depth - first_depth

    def _compute_proxemic_vector(self, bbox_delta, depth_delta):
        # Heuristic combination
        # bbox_delta is in % (e.g. 24.5 = 24.5% increase in area) -> Approach
        # depth_delta: -0.32 (decreasing depth) -> Approach
        
        # Normalize bbox: let's say +/- 50% is +/- 1.0
        norm_bbox = max(-1.0, min(1.0, bbox_delta / 50.0))
        
        # The prompt says: "A rapidly decreasing depth map value indicates approach."
        # So decreasing depth -> approach -> positive vector.
        # depth_delta = last - first. If decreasing, depth_delta < 0.
        # So norm_depth = -depth_delta * 2.0 (scaling factor)
        norm_depth = max(-1.0, min(1.0, -depth_delta * 2.0))
        vector = (norm_bbox * 0.4) + (norm_depth * 0.6)
            
        # Ignore +/- 0.05 micro-movements
        if abs(vector) < 0.05:
            vector = 0.0
            
        # Classify
        if vector > 0.3:
            action = "Approach_Intervention"
        elif vector < -0.3:
            action = "Avoidance"
        else:
            action = "Neutral"
            
        return vector, action
