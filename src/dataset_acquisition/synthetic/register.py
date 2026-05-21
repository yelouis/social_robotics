import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset_acquisition.registry import scan_datasets, save_registry

def register_synthetic_videos():
    print("[SyntheticRegister] Initiating dataset registration sweep...")
    registry_data = scan_datasets()
    save_registry(registry_data)
    print("[SyntheticRegister] Registry update completed successfully.")

if __name__ == "__main__":
    register_synthetic_videos()
