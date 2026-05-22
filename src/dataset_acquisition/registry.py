import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

REGISTRY_FILE = config.BASE_DIR / "local_video_registry.json"

def scan_datasets():
    registry = []
    seen_paths = set()
    
    for dataset_name, paths in config.DATASET_PATHS.items():
        print(f"Scanning for {dataset_name}...")
        for path in paths:
            if not path.exists():
                continue
            
            # Find all mp4 files recursively
            for video_file in path.rglob("*.mp4"):
                # Skip hidden files
                if video_file.name.startswith("._"):
                    continue

                abs_path = str(video_file.resolve())
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)
                
                # Use filename as ID 
                video_id = video_file.stem
                
                # Basic metadata
                stats = video_file.stat()
                
                entry = {
                    "id": video_id,
                    "dataset": dataset_name,
                    "file_path": str(video_file),
                    "file_size": stats.st_size,
                    "created_at": stats.st_ctime
                }
                if dataset_name == "synthetic_validation":
                    scenario_tag = video_file.parent.name
                    prompt_hash = video_file.stem
                    # Provenance (generator/version) is read from the per-clip
                    # sidecar written by generator.py, keeping the registry
                    # generator-agnostic. Fall back to sane defaults if absent.
                    generator = "wan2.1-t2v"
                    generator_version = "unknown"
                    sidecar = video_file.with_suffix(".json")
                    if sidecar.exists():
                        try:
                            with open(sidecar, "r") as sf:
                                meta = json.load(sf)
                            generator = meta.get("generator", generator)
                            generator_version = meta.get("generator_version", generator_version)
                        except (json.JSONDecodeError, OSError):
                            pass
                    entry.update({
                        "id": f"synthetic_{scenario_tag}_{prompt_hash}",
                        "synthetic": True,
                        "scenario_tag": scenario_tag,
                        "prompt_hash": prompt_hash,
                        "generator": generator,
                        "generator_version": generator_version,
                        "expected_pass": True
                    })
                registry.append(entry)
                
    print(f"Found {len(registry)} total videos across all datasets.")
    return registry

def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)
    print(f"Registry saved to {REGISTRY_FILE}")

if __name__ == "__main__":
    registry_data = scan_datasets()
    save_registry(registry_data)
