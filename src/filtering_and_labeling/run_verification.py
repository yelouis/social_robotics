import os
import sys
import argparse
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from filtering_and_labeling.pipeline import FilteringPipeline
import config

def run_verification(force=False):
    test_registry_path = config.BASE_DIR / "test_video_registry.json"
    
    # Ensure output directory exists on SSD
    config.ensure_dirs()
    output_manifest_path = config.OUTPUT_DIR / "filtered_manifest.json"
    
    if not test_registry_path.exists():
        print(f"Error: {test_registry_path} not found. Run dataset_acquisition/run_test_batch.py first.")
        return
        
    pipeline = FilteringPipeline(test_registry_path, output_manifest_path, force=force)
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run filtering and labeling verification.")
    parser.add_argument("--force", action="store_true", help="Force re-processing of already processed videos.")
    args = parser.parse_args()
    
    run_verification(force=args.force)
