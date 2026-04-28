import json
import cv2
import torch
import numpy as np
import tempfile
import os
import shutil
import traceback
from pathlib import Path
from ultralytics import YOLO
import ollama
from shared.social_presence import SocialPresenceDetector

class FilteringPipeline:
    def __init__(self, input_manifest_path, output_manifest_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_manifest_path = Path(output_manifest_path)
        self.error_log_path = self.output_manifest_path.parent / "02_filter_errors.json"
        self.force = force
        
        # Load YOLOv8 model via shared detector
        print("Loading YOLO model...")
        self.detector = SocialPresenceDetector('yolov8n.pt')
        self.vlm_model = "qwen2.5vl"
        
        self.processed_ids = set()
        if self.output_manifest_path.exists() and not self.force:
            try:
                with open(self.output_manifest_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing manifest: {e}. Starting fresh.")
        
    def run(self):
        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)
            
        filtered_results = []
        # Load existing results to append to them if not forcing
        if self.output_manifest_path.exists() and not self.force:
            try:
                with open(self.output_manifest_path, 'r') as f:
                    filtered_results = json.load(f)
            except:
                pass

        for entry in registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue

            print(f"Processing video: {video_id}")
            try:
                result = self.process_video(entry)
                if result:
                    filtered_results.append(result)
                    self.processed_ids.add(video_id)
                    
                    # Incremental save
                    with open(self.output_manifest_path, 'w') as f:
                        json.dump(filtered_results, f, indent=4)
            except Exception as e:
                self.log_error(video_id, e)
                
        print(f"Final count: {len(filtered_results)} videos in manifest.")

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

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['file_path'])
        if not video_path.exists():
            print(f"File not found: {video_path}")
            return None
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0 or total_frames == 0:
            cap.release()
            return None
            
        duration_sec = total_frames / fps
        
        # 1. Social Presence Filter (1 FPS)
        bystander_detections = self.social_presence_filter(video_path)
        if not bystander_detections:
            print(f"Dropped {video_id}: No bystanders detected.")
            cap.release()
            return None
            
        # 2. Contextual Task Labeling & Velocity (Every 5 seconds)
        identified_tasks = self.contextual_task_labeling(cap, fps, total_frames, duration_sec)
        if not identified_tasks:
            print(f"Dropped {video_id}: No clear tasks identified.")
            cap.release()
            return None
            
        # 3. Temporal Task Climax Identification
        self.temporal_climax_identification(cap, fps, total_frames, identified_tasks)
        
        cap.release()
        
        return {
            "video_id": video_id,
            "source_dataset": entry['dataset'],
            "video_path": str(video_path),
            "fps": fps,
            "duration_sec": duration_sec,
            "identified_tasks": identified_tasks,
            "bystander_detections": bystander_detections
        }

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

    def contextual_task_labeling(self, cap, fps, total_frames, duration_sec, interval_sec=5):
        """ Use Qwen2.5-VL to classify tasks """
        tasks = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for time_sec in range(0, int(duration_sec), interval_sec):
                frame_idx = int(time_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                img_path = temp_path / f"frame_{frame_idx}.jpg"
                cv2.imwrite(str(img_path), frame)
                
                prompt = (
                    "What task is the person performing in this egocentric/first-person video? "
                    "Respond with a short phrase describing the task. "
                    "If there is no clear task being performed, just say 'Idling'. "
                    "Also, estimate the physical velocity of the task as one of: [fast, medium, slow]. "
                    "Format your response EXACTLY as: 'Task: <task_description>. Velocity: <velocity>.'"
                )
                
                try:
                    response = ollama.chat(model=self.vlm_model, messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [str(img_path)]
                        }
                    ])
                    content = response['message']['content'].strip()
                    print(f"Raw VLM response: {content}")
                    
                    # Parse response
                    task_label = "Idling"
                    velocity = "medium"
                    
                    if "Task:" in content:
                        parts = content.split("Velocity:")
                        task_label = parts[0].replace("Task:", "").strip().strip('.')
                        if len(parts) > 1:
                            v = parts[1].strip().lower().strip('.')
                            if v in ['fast', 'medium', 'slow']:
                                velocity = v
                    
                    # Normalize labels
                    normalized_label = task_label.lower().strip().strip('.')
                    if not normalized_label or "idling" in normalized_label or "no clear task" in normalized_label:
                        task_label = "Idling"
                    
                    tasks.append({
                        "start_sec": time_sec,
                        "end_sec": min(time_sec + interval_sec, duration_sec),
                        "label": task_label,
                        "velocity": velocity
                    })
                    
                except Exception as e:
                    print(f"Ollama error: {e}")
                    
        # Merge identical sequential tasks
        merged_tasks = []
        current_task = None
        merge_count = 0
        
        for t in tasks:
            label = t['label']
            if label.lower().strip('.') in ["idling", "no clear task"]:
                continue
                
            # Normalize for comparison
            normalized_label = label.lower().strip().strip('.')
            
            if current_task is None:
                current_task = {
                    "task_id": f"t_{len(merged_tasks)+1:02d}",
                    "task_label": label,
                    "task_confidence": 0.0, # Placeholder
                    "task_velocity": t['velocity'],
                    "task_start_sec": t['start_sec'],
                    "task_end_sec": t['end_sec'],
                    "task_temporal_metadata": {}
                }
                merge_count = 1
            elif current_task['task_label'].lower().strip().strip('.') == normalized_label:
                current_task['task_end_sec'] = t['end_sec']
                merge_count += 1
            else:
                # Calculate confidence before switching
                current_task['task_confidence'] = round(min(0.95, 0.7 + (merge_count * 0.05)), 2)
                merged_tasks.append(current_task)
                current_task = {
                    "task_id": f"t_{len(merged_tasks)+1:02d}",
                    "task_label": label,
                    "task_confidence": 0.0,
                    "task_velocity": t['velocity'],
                    "task_start_sec": t['start_sec'],
                    "task_end_sec": t['end_sec'],
                    "task_temporal_metadata": {}
                }
                merge_count = 1
                
        if current_task:
            current_task['task_confidence'] = round(min(0.95, 0.7 + (merge_count * 0.05)), 2)
            merged_tasks.append(current_task)
            
        return merged_tasks

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
                
            climax_sec = climax_frame / fps
            extraction_method = "optical_flow_peak"
            vlm_confidence = None
            
            # Stage 2: VLM Refinement for slow tasks
            if task['task_velocity'] == 'slow' and len(flow_data) > 1:
                # Sort by flow and pick top 3 candidates around the peak
                candidates = sorted(flow_data, key=lambda x: x[1], reverse=True)[:3]
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    best_vlm_score = -1
                    refined_climax_frame = climax_frame
                    
                    for cand_frame, cand_mag in candidates:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cand_frame)
                        ret, frame = cap.read()
                        if not ret: continue
                        
                        img_path = temp_path / f"cand_{cand_frame}.jpg"
                        cv2.imwrite(str(img_path), frame)
                        
                        prompt = (
                            f"The person is performing the task: '{task['task_label']}'. "
                            "On a scale of 0 to 10, how well does this image represent the 'climax' or "
                            "the most critical moment of this action? Respond with just the number."
                        )
                        
                        try:
                            response = ollama.chat(model=self.vlm_model, messages=[
                                {'role': 'user', 'content': prompt, 'images': [str(img_path)]}
                            ])
                            score_str = response['message']['content'].strip()
                            # Extract first number
                            import re
                            scores = re.findall(r'\d+', score_str)
                            if scores:
                                score = int(scores[0])
                                if score > best_vlm_score:
                                    best_vlm_score = score
                                    refined_climax_frame = cand_frame
                        except:
                            pass
                    
                    if best_vlm_score != -1:
                        climax_frame = refined_climax_frame
                        climax_sec = climax_frame / fps
                        extraction_method = "optical_flow_peak+vlm_refinement"
                        vlm_confidence = round(best_vlm_score / 10.0, 2)

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
