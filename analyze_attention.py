import argparse
import json
import cv2
import os
import math
import numpy as np
import torch
from l2cs import Pipeline

def get_attention_score(pitch, yaw):
    """
    Converts Euler angles (pitch, yaw) to an attention score [0, 1].
    A score of 1.0 means looking dead straight into the camera lens.
    We assume max angle is roughly 1.0 radians (~57 degrees), beyond which attention is 0.
    """
    max_angle = 1.0 
    angle_off_center = max(abs(pitch), abs(yaw))
    score = max(0.0, 1.0 - (float(angle_off_center) / max_angle))
    return score

def run_attention_analysis(video_path, output_json, weights_path="models/L2CSNet_gaze360.pkl", sampling_rate_sec=0.5):
    """
    Processes a video with L2CS-Net to track 3D gaze and calculate attention scores.
    """
    print(f"Processing VIDEO: {video_path}")
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at {weights_path}.")
        print("Please download L2CSNet_gaze360.pkl and place it in the models/ directory.")
        return
    
    # 1. Setup Device 
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 2. Setup SOTA L2CS Gaze Pipeline
    print("Loading L2CS-Net SOTA Gaze Pipeline...")
    gaze_pipeline = Pipeline(
        weights=weights_path,
        arch='ResNet50',
        device=device
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0
        
    frame_stride = max(1, int(fps * sampling_rate_sec))
    
    attention_trace = []
    
    frame_idx = 0
    t_sec = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_stride == 0:
            t_sec = frame_idx / fps
            
            # Convert BGR to RGB for CNN inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run Gaze Estimation Pipeline (Includes face detection + ResNet Gaze regression)
            results = gaze_pipeline.step(rgb_frame)
            
            if results and hasattr(results, 'pitch') and len(results.pitch) > 0:
                # For simplicity, if multiple faces, take the one with highest score or centrally located
                # In a full pipeline, we map results.bboxes to `bystander_detections` body box.
                
                # Assuming index 0 is our best tracked person for this mock processing
                best_pitch = float(results.pitch[0])
                best_yaw = float(results.yaw[0])
                
                score = get_attention_score(best_pitch, best_yaw)
                attention_trace.append({
                    "t": round(t_sec, 2),
                    "score": round(score, 3),
                    "pitch_rad": round(best_pitch, 4),
                    "yaw_rad": round(best_yaw, 4)
                })
            else:
                # No face detected
                attention_trace.append({"t": round(t_sec, 2), "score": 0.0, "pitch_rad": None, "yaw_rad": None})
                
        frame_idx += 1

    cap.release()
    
    # 3. Aggregate Data
    if len(attention_trace) > 0:
        avg_score = sum(x["score"] for x in attention_trace) / len(attention_trace)
        peak_t = max(attention_trace, key=lambda x: x["score"])["t"]
        scores = [x["score"] for x in attention_trace]
        variance = np.var(scores) if len(scores) > 1 else 0.0
    else:
        avg_score = 0.0
        peak_t = 0.0
        variance = 0.0
        
    is_engaged = avg_score > 0.6  # Threshold
    
    # 4. Save to Output Schema as defined in 03a_attention_layer.md
    output_data = {
        "video_id": os.path.basename(video_path),
        "layer": "03a_attention",
        "processing_meta": {
            "model_used": "l2cs_net_3d_gaze",
            "sampling_fps_effective": fps / frame_stride,
            "device": str(device)
        },
        "per_person": [
            {
                "person_id": 0,
                "average_attention_score": round(avg_score, 3),
                "peak_engagement_timestamp_sec": peak_t,
                "attention_variance": round(float(variance), 4),
                "sustained_engagement_sec": 0.0, # Needs advanced sequential logic
                "is_engaged": bool(is_engaged),
                "gaze_target_classification": "POV_Actor" if is_engaged else "Other",
                "attention_trace": attention_trace
            }
        ],
        "aggregate": {
            "num_bystanders_tracked": 1,
            "mean_attention_all_persons": round(avg_score, 3),
            "any_person_engaged": bool(is_engaged)
        }
    }
    
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Attention analysis complete! Saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA 3D Gaze Estimation for Node 03a")
    parser.add_argument("--video-path", type=str, required=True, help="Path to input video clip")
    parser.add_argument("--output-json", type=str, default="03a_attention_result.json", help="Path to output manifest")
    parser.add_argument("--weights-path", type=str, default="models/L2CSNet_gaze360.pkl", help="Path to L2CS-Net weights")
    parser.add_argument("--stride-sec", type=float, default=0.5, help="Sampling stride in seconds")
    
    args = parser.parse_args()
    
    run_attention_analysis(args.video_path, args.output_json, args.weights_path, args.stride_sec)
