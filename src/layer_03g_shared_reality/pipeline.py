
import json
import traceback
import cv2
import numpy as np
from pathlib import Path

class SharedRealityPipeline:
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

        print(f"Final count: {len(results)} videos processed for Shared Reality.")

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
            
            # Step 1: Camera Shift extraction using optical flow
            shift_vector = self._extract_camera_shift(video_path, start_sec, end_sec)
            
            # Step 2: Bystander centering logic
            bystander_centered = self._check_bystander_centering(
                video_path, bystanders, start_sec, end_sec
            )
            
            # Social referencing requires BOTH a meaningful camera shift
            # (indicating the POV actor looked away from the task) AND
            # the bystander ending up centered in the FOV.
            shift_magnitude = np.sqrt(shift_vector[0]**2 + shift_vector[1]**2)
            social_reference_sought = bool(bystander_centered and shift_magnitude > 30.0)
            
            tasks_analyzed.append({
                "task_id": task_id,
                "post_climax_camera_shift_vector": shift_vector,
                "bystander_centered_in_fov": bystander_centered,
                "social_reference_sought": social_reference_sought
            })
                
        if not tasks_analyzed:
            return None
            
        return {
            "video_id": video_id,
            "layer": "03g_shared_reality",
            "tasks_analyzed": tasks_analyzed
        }

    def _extract_camera_shift(self, video_path, start_sec, end_sec):
        """
        Uses Farneback optical flow to estimate the global camera shift during the reaction window.
        Returns the accumulated shift vector [dx, dy].
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [0.0, 0.0]
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            return [0.0, 0.0]
            
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return [0.0, 0.0]
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (0, 0), fx=0.25, fy=0.25) # Downsample for speed
        
        current_frame_idx = start_frame + 1
        
        total_dx = 0.0
        total_dy = 0.0
        
        while current_frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # The flow vector at each pixel indicates where that pixel moved. 
            # The average flow is an estimation of background movement (assuming background is dominant).
            # To get camera shift, we take the negative of the background movement.
            # E.g. if the background moves left (-x), the camera panned right (+x).
            mean_dx = -np.mean(flow[..., 0])
            mean_dy = -np.mean(flow[..., 1])
            
            # Upscale the shift back to original resolution scale
            total_dx += mean_dx * 4.0
            total_dy += mean_dy * 4.0
            
            prev_gray = gray
            current_frame_idx += 1
            
        cap.release()
        
        return [round(total_dx, 2), round(total_dy, 2)]

    def _check_bystander_centering(self, video_path, bystanders, start_sec, end_sec):
        """
        Checks if any bystander's bounding box centroid falls within the middle 30% of the frame
        during the reaction window.
        """
        if not bystanders:
            return False
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
            
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        
        if width == 0 or height == 0:
            return False
            
        min_x = width * 0.35
        max_x = width * 0.65
        min_y = height * 0.35
        max_y = height * 0.65
        
        for bystander in bystanders:
            timestamps = bystander.get('timestamps_sec', [])
            bboxes = bystander.get('bounding_boxes', [])
            
            if len(timestamps) != len(bboxes):
                continue
                
            for t, bbox in zip(timestamps, bboxes):
                if start_sec <= t <= end_sec:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    if min_x <= cx <= max_x and min_y <= cy <= max_y:
                        return True
                        
        return False
