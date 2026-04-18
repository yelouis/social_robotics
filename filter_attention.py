import pandas as pd
import json
import os

def load_manifest(filepath):
    """Reads the paired clips manifest CSV into a Pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"Error: Manifest file not found at {filepath}")
        return None
    return pd.read_csv(filepath)

def merge_intervals(intervals):
    """Merges overlapping intervals and returns the total covered duration."""
    if not intervals:
        return 0, []
    
    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = []
    if not sorted_intervals:
        return 0, []
    
    curr_start, curr_end = sorted_intervals[0]
    
    for next_start, next_end in sorted_intervals[1:]:
        if next_start <= curr_end:
            # Overlap, extend the current interval
            curr_end = max(curr_end, next_end)
        else:
            # No overlap, push the current interval and start a new one
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
            
    merged.append((curr_start, curr_end))
    
    total_duration = sum(end - start for start, end in merged)
    return total_duration, merged

def calculate_activity_coverage(actions_str, clip_length_s):
    """
    Parses actions string and computes activity fraction.
    actions_str: "class start end;class start end;..."
    """
    if pd.isna(actions_str) or not actions_str:
        return 0.0, None
    
    intervals = []
    triplets = actions_str.split(';')
    for triplet in triplets:
        parts = triplet.strip().split(' ')
        if len(parts) == 3:
            try:
                # class_id = parts[0]
                t_start = float(parts[1])
                t_end = float(parts[2])
                intervals.append((t_start, t_end))
            except ValueError:
                continue
    
    total_covered, merged = merge_intervals(intervals)
    if clip_length_s is None or pd.isna(clip_length_s) or clip_length_s <= 0:
        activity_fraction = 0.0
    else:
        activity_fraction = total_covered / clip_length_s
    
    # Find the longest contiguous annotated segment (primary window)
    primary_window = {"t_start": 0.0, "t_end": 0.0}
    if merged:
        longest_interval = max(merged, key=lambda x: x[1] - x[0])
        primary_window = {"t_start": longest_interval[0], "t_end": longest_interval[1]}
        
    return activity_fraction, primary_window

def evaluate_attention(row, min_threshold=0.20):
    """Evaluates if a clip passes the attention (activity) threshold."""
    actions_str = row['actions']
    clip_length = row['length']
    
    activity_fraction, primary_window = calculate_activity_coverage(actions_str, clip_length)
    
    passed = activity_fraction >= min_threshold
    
    return {
        "ego_video_id": row['ego_video_id'],
        "tp_video_id": row['tp_video_id'],
        "passed_attention": passed,
        "activity_fraction": round(activity_fraction, 4),
        "primary_window": primary_window
    }

def main():
    # Configuration
    DATA_DIR = "/Volumes/Extreme SSD/charades_ego_data"
    ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations", "CharadesEgo")
    MANIFEST_PATH = os.path.join(ANNOTATIONS_DIR, "paired_clips_manifest.csv")
    OUTPUT_PATH = "attention_results.json"
    
    print(f"Loading manifest from {MANIFEST_PATH}...")
    df = load_manifest(MANIFEST_PATH)
    if df is None:
        return
    
    print(f"Processing {len(df)} clips...")
    results = []
    for _, row in df.iterrows():
        result = evaluate_attention(row)
        results.append(result)
    
    print(f"Saving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    passed_count = sum(1 for r in results if r['passed_attention'])
    print(f"Filtering complete.")
    print(f"Total clips: {len(df)}")
    print(f"Passed attention filter: {passed_count} ({passed_count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
