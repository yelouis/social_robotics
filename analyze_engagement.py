import os
import json
import cv2
import pandas as pd
import ollama
import base64
import argparse

def load_data(manifest_path, attention_path):
    """Loads manifest and attention results."""
    manifest_df = pd.read_csv(manifest_path)
    with open(attention_path, 'r') as f:
        attention_results = json.load(f)
    
    # Filter for passed clips
    passed_clips = [c for c in attention_results if c['passed_attention']]
    
    return manifest_df, passed_clips

def sample_frames_base64(video_path, t_start, t_end, fps=1.0, max_frames=5):
    """Samples frames and returns them as a list of base64 strings."""
    if not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    duration = t_end - t_start
    sample_interval = 1.0 / fps
    num_samples = min(int(duration * fps), max_frames)
    
    base64_frames = []
    for i in range(num_samples):
        timestamp = t_start + (i * sample_interval)
        if timestamp > t_end:
            break
            
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if ret:
            # Resize for speed and memory efficiency
            frame = cv2.resize(frame, (640, 480))
            # Encode as JPG
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
        else:
            break
            
    cap.release()
    return base64_frames

def generate_engagement_prompt():
    """Returns the strict-JSON formatting prompt for cognitive state classification."""
    return (
        "Observe the person in these video frames. Classify their cognitive state "
        "into exactly one category: [Focused, Neutral, Startled]. "
        "Focused: Attentive and deliberate on a task. "
        "Neutral: Relaxed or unremarkable expression. "
        "Startled: Show surprise, flinching, or sudden reactive movement. "
        "Return ONLY a JSON object: {\"state\": \"...\", \"confidence\": 0.0}"
    )

def analyze_engagement(video_path, t_start, t_end):
    """Runs VLM inference via Ollama using the chat() API for proper base64 image support."""
    frames = sample_frames_base64(video_path, t_start, t_end)
    if not frames:
        return "UNKNOWN", 0.0
    
    prompt = generate_engagement_prompt()
    
    try:
        response = ollama.chat(
            model='moondream',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': frames
            }],
            format='json'
        )
        
        data = json.loads(response['message']['content'])
        state = data.get("state", "UNKNOWN").upper()
        # Default to 1.0 for successful parses if not provided by VLM
        confidence = data.get("confidence", 1.0)
        
        valid_states = ["FOCUSED", "NEUTRAL", "STARTLED"]
        if state not in valid_states:
            # Try fuzzy matching or extraction
            for v in valid_states:
                if v in state:
                    return v, confidence
            
            print(f"  [LOG] Hallucination/Unmapped: '{state}' recorded as UNKNOWN")
            state = "UNKNOWN"
            confidence = 0.3
            
        return state, confidence
        
    except Exception as e:
        print(f"  [LOG] Ollama/JSON error: {e}. Assigned UNKNOWN.")
        return "UNKNOWN", 0.3

def main():
    parser = argparse.ArgumentParser(description="Module 03: Engagement VLM Analysis via Ollama")
    parser.add_argument('--limit', type=int, default=None,
                        help="Max number of clips to process (default: all passing clips)")
    args = parser.parse_args()
    
    # Configuration
    DATA_DIR = "/Volumes/Extreme SSD/charades_ego_data"
    MANIFEST_PATH = os.path.join(DATA_DIR, "annotations", "CharadesEgo", "paired_clips_manifest.csv")
    ATTENTION_PATH = "attention_results.json"
    OUTPUT_PATH = "engagement_results.json"
    
    print(f"Loading data...")
    manifest_df, passed_clips = load_data(MANIFEST_PATH, ATTENTION_PATH)
    
    clips_to_process = passed_clips[:args.limit] if args.limit else passed_clips
    total = len(clips_to_process)
    
    results = []
    print(f"Processing {total} clips via Ollama (Moondream)...")
    
    for i, clip in enumerate(clips_to_process):
        ego_id = clip['ego_video_id']
        tp_id = clip['tp_video_id']
        t_start = clip['primary_window']['t_start']
        t_end = clip['primary_window']['t_end']
        
        row = manifest_df[manifest_df['ego_video_id'] == ego_id].iloc[0]
        tp_video_path = row['tp_video_path']
        
        print(f"[{i+1}/{total}] Analyzing {tp_id} ({t_start}s - {t_end}s)...")
        state, confidence = analyze_engagement(tp_video_path, t_start, t_end)
        
        results.append({
            "ego_video_id": ego_id,
            "tp_video_id": tp_id,
            "state": state,
            "t_start": t_start,
            "t_end": t_end,
            "confidence": confidence
        })
        print(f"  Result: {state} (conf: {confidence})")
        
    print(f"Saving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Module 03 analysis complete. {total} clips processed.")

if __name__ == "__main__":
    main()
