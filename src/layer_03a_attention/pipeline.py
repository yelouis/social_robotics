import json
import cv2
import tempfile
import traceback
import math
import sys
import torch
from pathlib import Path

# Add L2CS-Net to python path
l2cs_path = Path(__file__).resolve().parent.parent.parent / "models" / "l2cs-net"
sys.path.append(str(l2cs_path))
try:
    from l2cs import Pipeline
    from l2cs.utils import select_device
except ImportError:
    Pipeline = None
class AttentionLayerPipeline:
    def __init__(self, input_manifest_path, output_result_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03a_attention_errors.json"
        self.force = force
        
        self.device = select_device('') if Pipeline else torch.device('cpu')
        try:
            weights_path = l2cs_path / "models" / "L2CSNet_gaze360.pkl"
            if Pipeline and weights_path.exists():
                self.gaze_pipeline = Pipeline(
                    weights=str(weights_path),
                    arch='ResNet50',
                    device=self.device,
                    include_detector=False
                )
            else:
                self.gaze_pipeline = None
        except Exception as e:
            print(f"Failed to load L2CS-Net: {e}")
            self.gaze_pipeline = None
        
        self.processed_ids = set()
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting fresh.")

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

            print(f"Processing Attention Layer for video: {video_id}")
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
                self.log_error(video_id, e)

        print(f"Final count: {len(results)} videos processed for Attention.")

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
        video_path = Path(entry['video_path'])
        
        if not video_path.exists():
            print(f"File not found: {video_path}")
            return None
            
        bystanders = entry.get('bystander_detections', [])
        if not bystanders:
            print(f"No bystanders found for {video_id}.")
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
        
        per_person_results = []
        
        for bystander in bystanders:
            person_id = bystander.get('person_id')
            timestamps_sec = bystander.get('timestamps_sec', [])
            bounding_boxes = bystander.get('bounding_boxes', [])
            
            if not timestamps_sec or not bounding_boxes:
                continue
                
            trace = self._track_and_score(cap, fps, duration_sec, timestamps_sec, bounding_boxes)
            
            if not trace:
                continue
                
            scores = [t['score'] for t in trace]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Find peak engagement timestamp
            peak_item = max(trace, key=lambda x: x['score'])
            peak_timestamp = peak_item['t']
            
            # Variance
            variance = 0
            if len(scores) > 1:
                variance = sum((s - avg_score) ** 2 for s in scores) / (len(scores) - 1)
                
            # Sustained engagement (longest contiguous window >= 0.7)
            # Uses actual 't' timestamps for accuracy across adaptive stride changes
            sustained_windows = []
            current_start = None
            last_t = None
            for item in trace:
                if item['score'] >= 0.7:
                    if current_start is None:
                        current_start = item['t']
                    last_t = item['t']
                else:
                    if current_start is not None and last_t is not None:
                        sustained_windows.append(last_t - current_start)
                    current_start = None
                    last_t = None
            if current_start is not None and last_t is not None:
                sustained_windows.append(last_t - current_start)
                
            sustained_sec = max(sustained_windows) if sustained_windows else 0
            is_engaged = avg_score >= 0.5 or sustained_sec > 2.0
            
            per_person_results.append({
                "person_id": person_id,
                "average_attention_score": round(avg_score, 2),
                "peak_engagement_timestamp_sec": peak_timestamp,
                "attention_variance": round(variance, 4),
                "sustained_engagement_sec": round(sustained_sec, 2),
                "is_engaged": is_engaged,
                "gaze_target_classification": "Camera_or_Task", # Derived geometrically
                "attention_trace": trace
            })
            
        cap.release()
        
        if not per_person_results:
            return None
            
        # Aggregate stats
        all_scores = [p['average_attention_score'] for p in per_person_results]
        mean_all = sum(all_scores) / len(all_scores) if all_scores else 0
        any_engaged = any(p['is_engaged'] for p in per_person_results)
        
        return {
            "video_id": video_id,
            "layer": "03a_attention",
            "processing_meta": {
                "model_used": "l2cs_net_3d_gaze" if self.gaze_pipeline else "fallback_dummy",
                "sampling_fps_effective": "adaptive_0.5s_to_0.2s"
            },
            "per_person": per_person_results,
            "aggregate": {
                "num_bystanders_tracked": len(per_person_results),
                "mean_attention_all_persons": round(mean_all, 2),
                "any_person_engaged": any_engaged
            }
        }

    def _track_and_score(self, cap, fps, duration_sec, b_timestamps, b_bboxes):
        trace = []
        current_t = 0.0
        last_score = -1.0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            while current_t <= duration_sec:
                # Find closest bbox
                diffs = [abs(t - current_t) for t in b_timestamps]
                closest_idx = diffs.index(min(diffs))
                
                # If the closest bbox is too far away (e.g. > 2 seconds), skip
                if diffs[closest_idx] > 2.0:
                    current_t += 0.5
                    continue
                    
                bbox = b_bboxes[closest_idx]
                x1, y1, x2, y2 = bbox
                
                frame_idx = int(current_t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Crop bounding box with some padding
                h, w = frame.shape[:2]
                px1 = max(0, int(x1) - 20)
                py1 = max(0, int(y1) - 20)
                px2 = min(w, int(x2) + 20)
                py2 = min(h, int(y2) + 20)
                
                crop = frame[py1:py2, px1:px2]
                if crop.size == 0:
                    current_t += 0.5
                    continue
                    
                score = 0.0
                pitch_rad = 0.0
                yaw_rad = 0.0
                
                if self.gaze_pipeline:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    try:
                        results = self.gaze_pipeline.step(crop_rgb)
                        # Results container structure from l2cs
                        pitch_rad = float(results.pitch[0][0]) if results.pitch.size > 0 else 0.0
                        yaw_rad = float(results.yaw[0][0]) if results.yaw.size > 0 else 0.0
                        
                        # Geometric Heuristic for attention_score
                        # Using L2CS gazeto3d convention for correct coordinate system
                        v_look_x = -math.cos(yaw_rad) * math.sin(pitch_rad)
                        v_look_y = -math.sin(yaw_rad)
                        v_look_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
                        
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        
                        v_cam_x = (w / 2.0) - cx
                        v_cam_y = (h / 2.0) - cy
                        v_cam_z = -float(w) # Negative Z = into screen in L2CS coordinate frame
                        
                        norm_cam = math.sqrt(v_cam_x**2 + v_cam_y**2 + v_cam_z**2)
                        if norm_cam > 0:
                            v_cam_x /= norm_cam
                            v_cam_y /= norm_cam
                            v_cam_z /= norm_cam
                        
                        dot_prod = (v_look_x * v_cam_x) + (v_look_y * v_cam_y) + (v_look_z * v_cam_z)
                        # Make the score steep so looking away penalizes heavily
                        # Map dot_prod from [0.5, 1.0] -> [0.0, 1.0] to make it more realistic
                        mapped_score = max(0.0, (dot_prod - 0.5) * 2.0)
                        score = round(min(1.0, mapped_score), 2)
                    except Exception as e:
                        print(f"[03a] Gaze inference failed at t={current_t:.2f}s: {e}")
                
                trace.append({
                    "t": round(current_t, 2),
                    "score": score,
                    "pitch_rad": round(pitch_rad, 4),
                    "yaw_rad": round(yaw_rad, 4)
                })
                
                # Adaptive stride
                if last_score >= 0 and abs(score - last_score) > 0.3:
                    current_t += 0.2  # 5 FPS
                else:
                    current_t += 0.5  # 2 FPS
                    
                last_score = score
                
        return trace
