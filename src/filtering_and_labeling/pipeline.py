import json
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import ollama
import tempfile
import os

class FilteringPipeline:
    def __init__(self, input_manifest_path, output_manifest_path):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_manifest_path = Path(output_manifest_path)
        
        # Load YOLOv8 model for person detection
        print("Loading YOLO model...")
        self.yolo_model = YOLO('yolov8n.pt')
        self.vlm_model = "qwen2.5vl"
        
    def run(self):
        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)
            
        filtered_results = []
        for entry in registry:
            print(f"Processing video: {entry['id']}")
            result = self.process_video(entry)
            if result:
                filtered_results.append(result)
                
        with open(self.output_manifest_path, 'w') as f:
            json.dump(filtered_results, f, indent=4)
        print(f"Filtered {len(filtered_results)} out of {len(registry)} videos.")
        print(f"Results saved to {self.output_manifest_path}")

    def process_video(self, entry):
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
        bystander_detections = self.social_presence_filter(cap, fps, total_frames)
        if not bystander_detections:
            print(f"Dropped {entry['id']}: No bystanders detected.")
            cap.release()
            return None
            
        # 2. Contextual Task Labeling & Velocity (Every 5 seconds)
        identified_tasks = self.contextual_task_labeling(cap, fps, total_frames, duration_sec)
        if not identified_tasks:
            print(f"Dropped {entry['id']}: No clear tasks identified.")
            cap.release()
            return None
            
        # 3. Temporal Task Climax Identification
        self.temporal_climax_identification(cap, fps, total_frames, identified_tasks)
        
        cap.release()
        
        return {
            "video_id": entry['id'],
            "source_dataset": entry['dataset'],
            "video_path": str(video_path),
            "fps": fps,
            "duration_sec": duration_sec,
            "identified_tasks": identified_tasks,
            "bystander_detections": bystander_detections
        }

    def social_presence_filter(self, cap, fps, total_frames, sample_rate_fps=1):
        """ Sample frames at sample_rate_fps and detect persons. """
        detections = {
            "person_id": 0,
            "timestamps_sec": [],
            "bounding_boxes": [],
            "detection_confidence": []
        }
        
        has_bystander = False
        frame_interval = int(max(1, fps / sample_rate_fps))
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_idx / fps
            
            # Run YOLO
            results = self.yolo_model(frame, classes=[0], verbose=False) # class 0 is person
            
            # Find the most confident person detection
            best_conf = 0
            best_box = None
            
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        # Format: [x1, y1, x2, y2]
                        best_box = [int(v) for v in box.xyxy[0].tolist()]
                        has_bystander = True
                        
            if best_box is not None:
                detections["timestamps_sec"].append(timestamp)
                detections["bounding_boxes"].append(best_box)
                detections["detection_confidence"].append(best_conf)
                
        if not has_bystander:
            return []
            
        return [detections]

    def contextual_task_labeling(self, cap, fps, total_frames, duration_sec, interval_sec=5):
        """ Use Moondream to classify tasks """
        tasks = []
        frame_interval = int(fps * interval_sec)
        
        temp_dir = Path(tempfile.mkdtemp())
        
        for time_sec in range(0, int(duration_sec), interval_sec):
            frame_idx = int(time_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            img_path = temp_dir / f"frame_{frame_idx}.jpg"
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
                
                # Clean up weird responses
                if not task_label or "idling" in task_label.lower() or "no clear task" in task_label.lower():
                    task_label = "Idling"
                

                
                tasks.append({
                    "start_sec": time_sec,
                    "end_sec": min(time_sec + interval_sec, duration_sec),
                    "label": task_label,
                    "velocity": velocity
                })
                
            except Exception as e:
                print(f"Ollama error: {e}")
                
            # Cleanup image
            if img_path.exists():
                img_path.unlink()
                
        temp_dir.rmdir()
        
        # Merge identical sequential tasks
        merged_tasks = []
        current_task = None
        
        for t in tasks:
            label = t['label']
            if label.lower().strip('.') in ["idling", "no clear task"]:
                continue
                
            if current_task is None:
                current_task = {
                    "task_id": f"t_{len(merged_tasks)+1:02d}",
                    "task_label": label,
                    "task_confidence": 0.85, # arbitrary default for now
                    "task_velocity": t['velocity'],
                    "start_sec": t['start_sec'],
                    "end_sec": t['end_sec'],
                    "task_temporal_metadata": {}
                }
            elif current_task['task_label'] == label:
                current_task['end_sec'] = t['end_sec']
            else:
                merged_tasks.append(current_task)
                current_task = {
                    "task_id": f"t_{len(merged_tasks)+1:02d}",
                    "task_label": label,
                    "task_confidence": 0.85,
                    "task_velocity": t['velocity'],
                    "start_sec": t['start_sec'],
                    "end_sec": t['end_sec'],
                    "task_temporal_metadata": {}
                }
        if current_task:
            merged_tasks.append(current_task)
            
        # We don't want start_sec/end_sec in final output, but we need them for optical flow
        return merged_tasks

    def temporal_climax_identification(self, cap, fps, total_frames, identified_tasks):
        """ Compute optical flow to find task climax """
        for task in identified_tasks:
            start_sec = task['start_sec']
            end_sec = task['end_sec']
            
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
                
                if mean_mag > max_flow:
                    max_flow = mean_mag
                    climax_frame = frame_idx
                    
                prev_gray = gray
                
            climax_sec = climax_frame / fps
            
            # Dynamic Reaction Window based on velocity
            velocity = task['task_velocity']
            if velocity == 'fast':
                window = [round(climax_sec + 0.5, 2), round(climax_sec + 2.0, 2)]
            elif velocity == 'medium':
                window = [round(climax_sec + 1.0, 2), round(climax_sec + 3.0, 2)]
            else: # slow or others
                window = [round(climax_sec + 2.0, 2), round(climax_sec + 6.0, 2)]
                
            task['task_temporal_metadata'] = {
                "task_climax_sec": round(climax_sec, 2),
                "task_reaction_window_sec": window,
                "climax_extraction_method": "optical_flow_peak",
                "optical_flow_peak_magnitude": round(float(max_flow), 2)
            }
            
            # Remove temporary fields
            del task['start_sec']
            del task['end_sec']

