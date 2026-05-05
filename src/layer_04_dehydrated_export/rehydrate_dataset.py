#!/usr/bin/env python3
"""
Rehydrate Dataset Script

This script demonstrates how a researcher can map the dehydrated social features
(from `social_metadata.parquet`) back onto their legally obtained copies of 
source videos (such as the Ego4D dataset).

Usage:
    python rehydrate_dataset.py --metadata social_metadata.parquet --videos /path/to/raw/videos
"""

import argparse
import pandas as pd
from pathlib import Path
import json

def rehydrate(metadata_path: str, raw_videos_dir: str):
    df = pd.read_parquet(metadata_path)
    videos_dir = Path(raw_videos_dir)
    
    print(f"Loaded {len(df)} metadata records.")
    
    matched_count = 0
    missing_count = 0
    
    for _, row in df.iterrows():
        video_id = row['video_id']
        # Typically, Ego4D videos are named <uuid>.mp4
        expected_video_path = videos_dir / f"{video_id}.mp4"
        
        if expected_video_path.exists():
            matched_count += 1
            print(f"Matched video: {video_id}")
            # Example of extracting trace data if 03a_attention ran
            if '03a_attention_per_person_raw' in row and pd.notna(row['03a_attention_per_person_raw']):
                try:
                    per_person = json.loads(row['03a_attention_per_person_raw'])
                    print(f"  - Features: Found {len(per_person)} people tracked.")
                except:
                    pass
        else:
            missing_count += 1
            print(f"Warning: Raw video not found for {video_id} at {expected_video_path}")
            
    print(f"\nRehydration Summary:")
    print(f"Total Records: {len(df)}")
    print(f"Matched Videos: {matched_count}")
    print(f"Missing Videos: {missing_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rehydrate social metadata with raw videos.")
    parser.add_argument("--metadata", required=True, help="Path to social_metadata.parquet")
    parser.add_argument("--videos", required=True, help="Path to directory containing raw .mp4 files")
    
    args = parser.parse_args()
    rehydrate(args.metadata, args.videos)
