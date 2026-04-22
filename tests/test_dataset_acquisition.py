import json
import os
from pathlib import Path
import cv2
import pytest

# Default registry file
DEFAULT_REGISTRY_FILE = Path(__file__).resolve().parent.parent / "local_video_registry.json"

@pytest.fixture
def registry():
    reg_file_path = os.getenv("REGISTRY_FILE")
    if reg_file_path:
        reg_file = Path(reg_file_path)
    else:
        reg_file = DEFAULT_REGISTRY_FILE
        
    if not reg_file.exists():
        pytest.skip(f"Registry file {reg_file} not found.")
        
    with open(reg_file, "r") as f:
        return json.load(f)

def test_batch_validation(registry):
    """Batch Test: Check if all files in registry exist and are non-empty."""
    assert len(registry) > 0, "Registry is empty"
    
    # Check a sample of files
    sample_size = min(len(registry), 500)
    for entry in registry[:sample_size]:
        path = Path(entry["file_path"])
        assert path.exists(), f"File not found: {path}"
        assert path.stat().st_size > 0, f"File is empty: {path}"
        assert path.suffix.lower() == ".mp4", f"Invalid file extension: {path}"

def test_singular_video_read(registry):
    """Singular Video Test: Try to open the first video and extract a frame."""
    if not registry:
        pytest.skip("No videos in registry")
    
    # Try the first few videos in case one is corrupt
    success = False
    for entry in registry[:3]:
        path = entry["file_path"]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
            
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            continue
            
        assert frame.shape[0] > 0 and frame.shape[1] > 0, "Invalid frame dimensions"
        cap.release()
        print(f"Successfully read frame from {path} (Size: {frame.shape})")
        success = True
        break
        
    assert success, "Could not read a frame from any of the first 3 videos"
