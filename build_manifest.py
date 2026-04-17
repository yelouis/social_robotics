import pandas as pd
import os
import glob

# Configuration
DATA_DIR = "/Volumes/Extreme SSD/charades_ego_data"
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations", "CharadesEgo")
EGO_VIDEOS_DIR = os.path.join(DATA_DIR, "ego_videos", "CharadesEgo_v1_480") # Tar extraction level
TP_VIDEOS_DIR = os.path.join(DATA_DIR, "tp_videos", "Charades_v1_480")      # Tar extraction level
OUTPUT_MANIFEST = os.path.join(ANNOTATIONS_DIR, "paired_clips_manifest.csv")

def build_manifest():
    print("Building paired clips manifest...")
    
    # Files to process
    train_csv = os.path.join(ANNOTATIONS_DIR, "CharadesEgo_v1_train.csv")
    test_csv = os.path.join(ANNOTATIONS_DIR, "CharadesEgo_v1_test.csv")
    
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"Error: Annotation CSVs not found in {ANNOTATIONS_DIR}")
        return

    # Load data
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Filter for egocentric clips
    # The CharadesEgo CSV format usually has an 'egocentric' column
    # and 'charades_video' column for the third-person pair.
    ego_df = df[df['egocentric'] == 'Yes'].copy()
    
    print(f"Found {len(ego_df)} egocentric clips.")
    
    # Prepare manifest data
    manifest_data = []
    
    for _, row in ego_df.iterrows():
        ego_id = row['id']
        tp_id = row['charades_video']
        
        # Construct paths
        ego_path = os.path.join(EGO_VIDEOS_DIR, f"{ego_id}.mp4")
        tp_path = os.path.join(TP_VIDEOS_DIR, f"{tp_id}.mp4")
        
        # Validation
        ego_exists = os.path.exists(ego_path)
        tp_exists = os.path.exists(tp_path)
        
        if ego_exists and tp_exists:
            manifest_data.append({
                'ego_video_id': ego_id,
                'ego_video_path': ego_path,
                'tp_video_id': tp_id,
                'tp_video_path': tp_path,
                'actions': row['actions'],
                'length': row['length'],
                'scene': row['scene'],
                'script': row['script']
            })
        else:
            if not ego_exists:
                print(f"Warning: Ego video missing: {ego_id}")
            if not tp_exists:
                print(f"Warning: TP video missing: {tp_id}")

    # Create manifest CSV
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(OUTPUT_MANIFEST, index=False)
    
    print(f"Manifest created with {len(manifest_df)} paired clips at {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    build_manifest()
