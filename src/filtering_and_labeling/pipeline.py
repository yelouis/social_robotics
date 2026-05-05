import json
import cv2
import gc
import torch
import numpy as np
import tempfile
import os
import shutil
import traceback
import re
from pathlib import Path
from ultralytics import YOLO
import ollama
from shared.social_presence import SocialPresenceDetector
import config


class FilteringPipeline:
    def __init__(self, input_manifest_path, output_manifest_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_manifest_path = Path(output_manifest_path)
        self.error_log_path = self.output_manifest_path.parent / "02_filter_errors.json"
        self.force = force
        
        # Shared components
        self.detector = SocialPresenceDetector('yolov8n.pt')
        self.vlm_model = "qwen2.5vl"
        
        # Pass storage
        self.initial_registry = [] # For Pass 1
        self.intermediate_results = [] # Pass 1 output
        
        self.processed_ids = set()
        if self.output_manifest_path.exists() and not self.force:
            try:
                with open(self.output_manifest_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing manifest: {e}. Starting fresh.")
        
        # Load Ego4D Metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """ Load Ego4D metadata and index by video_uid """
        metadata_path = config.EGO4D_METADATA_PATH
        if metadata_path.exists():
            print(f"Loading Ego4D metadata from {metadata_path}...")
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    videos = data.get('videos', [])
                    return {v['video_uid']: v for v in videos}
            except Exception as e:
                print(f"Error loading metadata: {e}")
        else:
            print(f"Metadata file not found at {metadata_path}")
        return {}

        
    def cleanup_yolo(self):
        """ Explicitly free YOLO memory before VLM pass """
        if hasattr(self, 'detector'):
            self.detector.unload()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def run(self):
        with open(self.input_manifest_path, 'r') as f:
            self.initial_registry = json.load(f)
            
        final_results = []
        # Load existing results to append to them if not forcing
        if self.output_manifest_path.exists() and not self.force:
            try:
                with open(self.output_manifest_path, 'r') as f:
                    final_results = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in final_results}
            except:
                pass

        # --- PASS 1: Social Filter (YOLO) ---
        print("\n--- PASS 1: Social Filter (YOLO) ---")
        pass1_queue = []
        for entry in self.initial_registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue
            
            video_path = Path(entry['file_path'])
            if not video_path.exists(): continue

            print(f"Social Filter: {video_id}")
            bystander_detections = self.social_presence_filter(video_path)
            if bystander_detections:
                entry['bystander_detections'] = bystander_detections
                pass1_queue.append(entry)
            else:
                print(f"Dropped {video_id}: No social presence.")

        # --- MEMORY CLEANUP ---
        self.cleanup_yolo()

        # --- PASS 2: Task Labeling & Climax (VLM) ---
        print("\n--- PASS 2: Task Labeling & Climax (VLM) ---")
        for entry in pass1_queue:
            video_id = entry.get('id', entry.get('video_id'))
            print(f"Task Refinement: {video_id}")
            try:
                result = self.process_video_vlm_pass(entry)
                if result:
                    final_results.append(result)
                    self.processed_ids.add(video_id)
                    
                    # Incremental save
                    with open(self.output_manifest_path, 'w') as f:
                        json.dump(final_results, f, indent=4)
            except Exception as e:
                self.log_error(video_id, e)
                
        print(f"Final count: {len(final_results)} videos in manifest.")

    def process_video_vlm_pass(self, entry):
        """ Pass 2: Labeling and Climax Identification """
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['file_path'])
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        
        # 2. Contextual Task Labeling
        identified_tasks = self.contextual_task_labeling(video_id, duration_sec)
        if not identified_tasks:
            cap.release()
            return None
            
        # 3. Temporal Task Climax Identification (Batched VLM)
        self.temporal_climax_identification(cap, fps, total_frames, identified_tasks)
        
        cap.release()
        
        return {
            "video_id": video_id,
            "source_dataset": entry['dataset'],
            "video_path": str(video_path),
            "fps": fps,
            "duration_sec": duration_sec,
            "identified_tasks": identified_tasks,
            "bystander_detections": entry['bystander_detections']
        }

    def log_error(self, video_id, error):
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
        with open(self.error_log_path, 'w') as f:
            json.dump(errors, f, indent=4)

    # No longer needed as we use two-pass
    # def process_video(self, entry): ...

    def social_presence_filter(self, video_path, sample_rate_fps=1):
        """ Sample frames at sample_rate_fps and detect all persons. """
        detections_by_frame = self.detector.detect(video_path, sample_rate_fps=sample_rate_fps, fast_mode=False)
        
        if not detections_by_frame:
            return []
            
        # Transform flat frame detections into the grouped bystander_detections schema
        # For now, since we don't have a cross-frame tracker, we'll treat all detections
        # in the video as potentially different people OR just group them by frame.
        # The schema requires:
        # { "person_id": X, "timestamps_sec": [], "bounding_boxes": [], "detection_confidence": [] }
        
        # Simple heuristic: If there are multiple people in a frame, they get different person_ids.
        # Without tracking, we can't easily link person_id 0 in frame A to person_id 0 in frame B.
        # But we MUST support multiple people.
        
        # To strictly follow the schema and support multi-person without a tracker:
        # We will create N entries in bystander_detections where N is the MAX number of people
        # seen in any single frame.
        max_people = max(len(frame) for frame in detections_by_frame)
        
        bystanders = []
        for i in range(max_people):
            bystanders.append({
                "person_id": i,
                "timestamps_sec": [],
                "bounding_boxes": [],
                "detection_confidence": []
            })
            
        for frame in detections_by_frame:
            for i, det in enumerate(frame):
                if i < max_people:
                    bystanders[i]["timestamps_sec"].append(det["timestamp_sec"])
                    bystanders[i]["bounding_boxes"].append(det["bounding_box"])
                    bystanders[i]["detection_confidence"].append(det["confidence"])
                    
        return bystanders

    def contextual_task_labeling(self, video_id, duration_sec):
        """ Use Ego4D metadata to identify tasks and map velocities """
        if video_id not in self.metadata:
            # Try removing extension if present in video_id
            clean_id = video_id.split('.')[0]
            if clean_id not in self.metadata:
                print(f"Warning: No metadata found for video {video_id}")
                return []
            video_meta = self.metadata[clean_id]
        else:
            video_meta = self.metadata[video_id]

        scenarios = video_meta.get('scenarios', [])
        if not scenarios:
            return []

        valid_scenarios = []
        for scenario in scenarios:
            label = scenario.strip()
            if label and label.lower() not in ["idling", "no clear task", "ambiguous activity"]:
                valid_scenarios.append(label)
        
        identified_tasks = []
        num_tasks = len(valid_scenarios)
        for i, label in enumerate(valid_scenarios):
            # Heuristic velocity mapping
            velocity = self._get_velocity_from_label(label)
            
            task_duration = duration_sec / num_tasks if num_tasks > 0 else duration_sec
            task_start_sec = i * task_duration
            task_end_sec = (i + 1) * task_duration
            
            task = {
                "task_id": f"t_{len(identified_tasks)+1:02d}",
                "task_label": label,
                "task_confidence": 1.0, # Ground truth metadata
                "task_velocity": velocity,
                "task_start_sec": round(task_start_sec, 2),
                "task_end_sec": round(task_end_sec, 2),
                "task_temporal_metadata": {}
            }
            identified_tasks.append(task)

        return identified_tasks

    def _get_velocity_from_label(self, label):
        """ Heuristic to map task labels to physical velocity """
        l = label.lower()
        fast_keywords = ['drop', 'slip', 'fall', 'throw', 'hit', 'impact', 'jump', 'run', 'fast', 'accident', 'shatter']
        slow_keywords = ['read', 'write', 'think', 'solve', 'puzzle', 'watch', 'look', 'sit', 'slow', 'wait', 'rest']
        
        if any(k in l for k in fast_keywords):
            return 'fast'
        if any(k in l for k in slow_keywords):
            return 'slow'
        return 'medium'


    def temporal_climax_identification(self, cap, fps, total_frames, identified_tasks):
        """ Compute optical flow to find task climax, with VLM refinement for slow tasks """
        for task in identified_tasks:
            start_sec = task['task_start_sec']
            end_sec = task['task_end_sec']
            
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            
            # Optical Flow params
            # Downsample temporal resolution to ~5 FPS for speed
            step = max(1, int(fps / 5))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, prev_frame = cap.read()
            if not ret:
                continue
                
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            # Resize for speed
            prev_gray = cv2.resize(prev_gray, (0,0), fx=0.5, fy=0.5)
            
            max_flow = 0.0
            climax_frame = start_frame
            flow_data = [] # Store candidate frames for VLM refinement
            
            # PASS 1: Coarse Pass at ~5 FPS
            for frame_idx in range(start_frame + step, end_frame, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
                
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = np.mean(mag)
                
                flow_data.append((frame_idx, mean_mag))
                
                if mean_mag > max_flow:
                    max_flow = mean_mag
                    climax_frame = frame_idx
                    
                prev_gray = gray
                
            # PASS 2: Dense Pass at Native FPS around Coarse Peak
            window_frames = int(1.0 * fps)
            dense_start = max(start_frame, climax_frame - window_frames)
            dense_end = min(end_frame, climax_frame + window_frames)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, dense_start)
            ret, prev_dense_frame = cap.read()
            if ret:
                prev_dense_gray = cv2.cvtColor(prev_dense_frame, cv2.COLOR_BGR2GRAY)
                prev_dense_gray = cv2.resize(prev_dense_gray, (0,0), fx=0.5, fy=0.5)
                
                dense_max_flow = 0.0
                dense_climax_frame = dense_start
                
                for frame_idx in range(dense_start + 1, dense_end):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, dense_frame = cap.read()
                    if not ret:
                        break
                        
                    dense_gray = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2GRAY)
                    dense_gray = cv2.resize(dense_gray, (0,0), fx=0.5, fy=0.5)
                    
                    flow = cv2.calcOpticalFlowFarneback(prev_dense_gray, dense_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mean_mag = np.mean(mag)
                    
                    if mean_mag > dense_max_flow:
                        dense_max_flow = mean_mag
                        dense_climax_frame = frame_idx
                        
                    prev_dense_gray = dense_gray
                
                climax_frame = dense_climax_frame
                max_flow = dense_max_flow if dense_max_flow > 0 else max_flow
                
            climax_sec = climax_frame / fps
            extraction_method = "optical_flow_peak"
            vlm_confidence = None
            
            # Stage 2: Batched VLM Refinement for slow tasks
            if task['task_velocity'] == 'slow' and len(flow_data) > 1:
                # Sort by flow and pick top 3 candidates around the peak
                candidates = sorted(flow_data, key=lambda x: x[1], reverse=True)[:3]
                candidates = sorted(candidates, key=lambda x: x[0]) # Chronological
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    img_paths = []
                    
                    for i, (cand_frame, _) in enumerate(candidates):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cand_frame)
                        ret, frame = cap.read()
                        if not ret: continue
                        
                        # Resize for VLM memory efficiency (max 1024px)
                        h, w = frame.shape[:2]
                        max_dim = 1024
                        if max(h, w) > max_dim:
                            scale = max_dim / max(h, w)
                            frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
                        
                        img_path = temp_path / f"cand_{i+1}.jpg"
                        cv2.imwrite(str(img_path), frame)
                        img_paths.append(str(img_path))
                    
                    if not img_paths: continue

                    prompt = (
                        f"The person is performing the task: '{task['task_label']}'. "
                        f"I have provided {len(img_paths)} images from the video. "
                        "Which image (respond with just the number 1, 2, or 3) "
                        "best represents the 'climax' or the most critical moment of this action? "
                        "If you are unsure, pick the one with the most active motion."
                    )
                    
                    try:
                        response = ollama.chat(model=self.vlm_model, messages=[
                            {'role': 'user', 'content': prompt, 'images': img_paths}
                        ])
                        content = response['message']['content'].strip()
                        matches = re.findall(r'[1-3]', content)
                        if matches:
                            choice = int(matches[0]) - 1 # 0-indexed
                            if choice < len(candidates):
                                climax_frame = candidates[choice][0]
                                climax_sec = climax_frame / fps
                                extraction_method = "optical_flow_peak+vlm_refinement"
                                # We treat the choice as "confidence 1.0" for the selected frame
                                vlm_confidence = 1.0 
                    except Exception as e:
                        print(f"VLM Refinement failed: {e}")
                        pass

            # Dynamic Reaction Window based on velocity
            velocity = task['task_velocity']
            if velocity == 'fast':
                window = [round(climax_sec + 0.5, 2), round(climax_sec + 2.0, 2)]
            elif velocity == 'medium':
                window = [round(climax_sec + 1.0, 2), round(climax_sec + 3.0, 2)]
            else: # slow
                window = [round(climax_sec + 2.0, 2), round(climax_sec + 6.0, 2)]
                
            task['task_temporal_metadata'] = {
                "task_climax_sec": round(climax_sec, 2),
                "task_reaction_window_sec": window,
                "climax_extraction_method": extraction_method,
                "optical_flow_peak_magnitude": round(float(max_flow), 2)
            }
            if vlm_confidence is not None:
                task['task_temporal_metadata']["vlm_climax_confidence"] = vlm_confidence
