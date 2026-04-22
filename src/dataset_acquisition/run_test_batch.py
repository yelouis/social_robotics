import os
import sys
import shutil
from pathlib import Path
import json

# Add src to path (the directory containing config.py and dataset_acquisition/)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from dataset_acquisition.downloader import CharadesEgoDownloader, Ego4DDownloader, EpicKitchensDownloader
from dataset_acquisition.registry import scan_datasets, save_registry

def run_test_batch(limit=5):
    print(f"--- Running Test Batch (Limit: {limit} videos per source) ---")
    
    # Create a test output directory
    test_output_dir = config.BASE_DIR / "test_raw_videos"
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporarily override OUTPUT_DIR in config
    original_output_dir = config.OUTPUT_DIR
    config.OUTPUT_DIR = test_output_dir
    
    datasets_to_test = ["charades_ego", "ego4d"]
    
    for ds_name in datasets_to_test:
        print(f"\nTesting {ds_name}...")
        ds_test_dir = test_output_dir / ds_name
        ds_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Find videos on SSD to copy as "samples"
        found_videos = []
        for p in config.DATASET_PATHS.get(ds_name, []):
            if p.exists():
                # ONLY copy files with size > 0 and NOT starting with ._
                videos = [v for v in p.rglob("*.mp4") if v.stat().st_size > 0 and not v.name.startswith("._")]
                if videos:
                    found_videos.extend(videos)
                    if len(found_videos) >= limit:
                        break
        
        if not found_videos:
            print(f"No valid (non-empty) videos found on SSD for {ds_name}. Skipping local copy test for this dataset.")
            continue
            
        samples = found_videos[:limit]
        print(f"Found {len(samples)} valid samples on SSD. Copying to test directory...")
        
        for v in samples:
            target = ds_test_dir / v.name
            if not target.exists():
                shutil.copy2(v, target)
            print(f"  Copied {v.name}")

    print("\nTesting epic_kitchens via official download script...")
    epic_dl = EpicKitchensDownloader()
    epic_dl.output_path = test_output_dir / "epic_kitchens"
    # Download 5 specific videos from P01
    test_epic_videos = ["P01_01", "P01_02", "P01_03", "P01_04", "P01_05"]
    epic_dl.download(specific_videos=test_epic_videos)

    # Now run the registry on the test directory
    original_paths = config.DATASET_PATHS.copy()
    datasets_to_test.append("epic_kitchens")
    for ds_name in datasets_to_test:
        config.DATASET_PATHS[ds_name] = [test_output_dir / ds_name]
    
    print("\nRunning Registry on test data...")
    registry = scan_datasets()
    
    # Save a test registry
    test_registry_file = config.BASE_DIR / "test_video_registry.json"
    with open(test_registry_file, "w") as f:
        json.dump(registry, f, indent=4)
    print(f"Test registry saved to {test_registry_file}")
    
    # Restore original config
    config.OUTPUT_DIR = original_output_dir
    config.DATASET_PATHS = original_paths
    
    print("\nTest batch complete.")
    print(f"Found {len(registry)} videos in test registry.")
    return test_registry_file

if __name__ == "__main__":
    run_test_batch()
