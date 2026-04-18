import os
import json
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Configuration
FLINCH_THRESHOLD = 150.0 # pixels/second
MANIFEST_PATH = "/Volumes/Extreme SSD/charades_ego_data/annotations/CharadesEgo/paired_clips_manifest.csv"
ENGAGEMENT_RESULTS = "engagement_results.json"
ATTENTION_RESULTS = "attention_results.json"
OUTPUT_FILE = "flinch_results.json"

# Initialize YOLOv8 Pose
# Using yolov8n-pose.pt (Nano) for low RAM footprint and high speed
print("Initializing YOLOv8-Pose model...")
model = YOLO('yolov8n-pose.pt')

def calculate_peak_velocity(pose_history, fps):
    """
    Calculates the peak velocity of upper-body landmarks.
    pose_history: list of numpy arrays (17, 3) -> [x, y, conf]
    """
    if len(pose_history) < 2:
        return 0.0, 0.0

    peak_v = 0.0
    total_conf = 0.0
    count = 0

    # COCO Keypoint Indices:
    # 0: nose, 5: l_shoulder, 6: r_shoulder, 9: l_wrist, 10: r_wrist, 11: l_hip, 12: r_hip
    upper_body_indices = [0, 5, 6, 9, 10, 11, 12]

    for i in range(1, len(pose_history)):
        prev_kp = pose_history[i-1]
        curr_kp = pose_history[i]
        
        max_delta = 0.0
        
        for idx in upper_body_indices:
            p1 = prev_kp[idx]
            p2 = curr_kp[idx]
            
            # Use confidence threshold (e.g., 0.3) to ensure reliable tracking
            if p1[2] > 0.3 and p2[2] > 0.3:
                # YOLOv8 keypoints.xy are in absolute pixels
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dist = np.sqrt(dx**2 + dy**2)
                
                max_delta = max(max_delta, dist)
                total_conf += p2[2]
                count += 1
        
        # Velocity in pixels/second: delta_pixels * fps
        velocity = max_delta * fps 
        peak_v = max(peak_v, velocity)

    avg_conf = total_conf / count if count > 0 else 0.0
    return peak_v, avg_conf

def process_clips():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest_map = manifest.set_index('ego_video_id').to_dict('index')

    print(f"Loading engagement results from {ENGAGEMENT_RESULTS}...")
    if not os.path.exists(ENGAGEMENT_RESULTS):
        print(f"Error: Engagement results not found.")
        return
    with open(ENGAGEMENT_RESULTS, 'r') as f:
        engagement_data = json.load(f)

    print(f"Loading attention results from {ATTENTION_RESULTS}...")
    if not os.path.exists(ATTENTION_RESULTS):
        print(f"Error: Attention results not found.")
        return
    with open(ATTENTION_RESULTS, 'r') as f:
        attention_data = json.load(f)
    attention_map = {item['ego_video_id']: item['passed_attention'] for item in attention_data}

    results = []

    for entry in engagement_data:
        ego_id = entry['ego_video_id']
        tp_id = entry['tp_video_id']
        t_start = entry['t_start']
        t_end = entry['t_end']

        # Node 1 Cross-check
        passed_attention = attention_map.get(ego_id, False)
        if not passed_attention:
            print(f"Skipping {ego_id}: Failed attention check.")
            continue

        if ego_id not in manifest_map:
            print(f"Warning: {ego_id} not in manifest. Skipping.")
            continue

        tp_video_path = manifest_map[ego_id]['tp_video_path']
        if not os.path.exists(tp_video_path):
            print(f"Error: Video not found: {tp_video_path}")
            continue

        print(f"Processing clip: {tp_id} ({t_start}s to {t_end}s)")
        cap = cv2.VideoCapture(tp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000)
        
        pose_history = []
        frame_count = 0
        max_frames = int((t_end - t_start) * fps)

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8-Pose inference
            # stream=True is more memory efficient for long videos
            inference_results = model(frame, verbose=False, device='cpu') # Use 'cpu' for stability, or 'mps' if supported
            
            if len(inference_results) > 0 and inference_results[0].keypoints is not None:
                # Check if any person was detected (data should have size > 0)
                if inference_results[0].keypoints.data.shape[0] > 0:
                    # Get the first person detected
                    kpts_data = inference_results[0].keypoints.data[0].cpu().numpy()
                    if kpts_data.shape == (17, 3):
                        pose_history.append(kpts_data)
            
            frame_count += 1
        
        cap.release()

        v_peak, confidence = calculate_peak_velocity(pose_history, fps)
        flinch = v_peak > FLINCH_THRESHOLD

        print(f"  Result: Peak V={v_peak:.2f} px/s, Flinch={flinch}, Conf={confidence:.2f}")

        results.append({
            "ego_video_id": ego_id,
            "tp_video_id": tp_id,
            "flinch": bool(flinch),
            "v_peak": round(float(v_peak), 2),
            "confidence": round(float(confidence), 2)
        })

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results for {len(results)} clips to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_clips()
