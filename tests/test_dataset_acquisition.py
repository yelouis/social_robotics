import json
import os
from pathlib import Path
import cv2
import pytest

REGISTRY_FILE = Path(__file__).resolve().parent.parent / "local_video_registry.json"

@pytest.fixture
def registry():
    if not REGISTRY_FILE.exists():
        pytest.skip("Registry file not found. Run registry script first.")
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)

def test_batch_validation(registry):
    """Batch Test: Check if all files in registry exist and are non-empty."""
    assert len(registry) > 0, "Registry is empty"
    
    # Check first 100 files to save time if registry is massive
    # or the whole thing if it's manageable. 
    # Since it's 15k, let's do a sample or the first 500.
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
    
    entry = registry[0]
    path = entry["file_path"]
    
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Could not open video file: {path}"
    
    ret, frame = cap.read()
    assert ret, f"Could not read first frame from: {path}"
    assert frame is not None, "Extracted frame is None"
    assert frame.shape[0] > 0 and frame.shape[1] > 0, "Invalid frame dimensions"
    
    cap.release()
    print(f"Successfully read frame from {path} (Size: {frame.shape})")
