import os
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from filtering_and_labeling.pipeline import FilteringPipeline
import config

def run_verification():
    test_registry_path = config.BASE_DIR / "test_video_registry.json"
    output_manifest_path = config.BASE_DIR / "filtered_manifest.json"
    
    if not test_registry_path.exists():
        print(f"Error: {test_registry_path} not found. Run dataset_acquisition/run_test_batch.py first.")
        return
        
    pipeline = FilteringPipeline(test_registry_path, output_manifest_path)
    pipeline.run()

if __name__ == "__main__":
    run_verification()
