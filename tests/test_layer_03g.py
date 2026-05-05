import os
import json
import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from src.layer_03g_shared_reality.pipeline import SharedRealityPipeline

@pytest.fixture
def dummy_manifest(tmp_path):
    manifest_path = tmp_path / "filtered_manifest.json"
    data = [
        {
            "video_id": "test_vid_01",
            "video_path": "nonexistent.mp4",
            "bystander_detections": [
                {
                    "person_id": 0,
                    "timestamps_sec": [1.0, 1.2, 1.4],
                    "bounding_boxes": [[10, 10, 50, 50], [10, 10, 50, 50], [10, 10, 50, 50]]
                }
            ],
            "identified_tasks": [
                {
                    "task_id": "t_01",
                    "task_temporal_metadata": {
                        "task_reaction_window_sec": [1.0, 2.0]
                    }
                }
            ]
        }
    ]
    with open(manifest_path, 'w') as f:
        json.dump(data, f)
    return manifest_path

@pytest.fixture
def pipeline_instance(dummy_manifest, tmp_path):
    output_path = tmp_path / "03g_output.json"
    return SharedRealityPipeline(dummy_manifest, output_path)

def _create_synthetic_video(path, fps=30, duration_sec=3.0, width=480, height=480, pan_velocity_x=0.0, pan_velocity_y=0.0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = int(fps * duration_sec)
    
    # We will simulate panning by translating a checkerboard pattern
    # Create a base image larger than the frame
    base_size = int(max(width, height) * 2)
    base_img = np.zeros((base_size, base_size, 3), dtype=np.uint8)
    for y in range(0, base_size, 40):
        for x in range(0, base_size, 40):
            if (x // 40 + y // 40) % 2 == 0:
                base_img[y:y+40, x:x+40] = 255
                
    center_x = base_size // 2
    center_y = base_size // 2

    for i in range(total_frames):
        # Calculate current top-left coordinate based on panning
        tx = int(pan_velocity_x * i)
        ty = int(pan_velocity_y * i)
        
        # We start looking at the center of the base image
        start_x = center_x - width // 2 + tx
        start_y = center_y - height // 2 + ty
        
        frame = base_img[start_y:start_y+height, start_x:start_x+width]
        out.write(frame)
    out.release()

def test_pipeline_initialization(dummy_manifest, tmp_path):
    output_path = tmp_path / "03g_output.json"
    pipeline = SharedRealityPipeline(dummy_manifest, output_path)
    assert pipeline is not None

def test_missing_video_handling(pipeline_instance, dummy_manifest):
    pipeline_instance.run()
    # Should gracefully skip non-existent videos
    assert not pipeline_instance.output_result_path.exists()

def test_camera_shift_static_video(pipeline_instance, tmp_path):
    video_path = tmp_path / "static.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=2.0)
    
    dx, dy = pipeline_instance._extract_camera_shift(video_path, 0.0, 1.5)
    
    # Assert shift is close to 0
    assert abs(dx) < 5.0
    assert abs(dy) < 5.0

def test_camera_shift_panning_video(pipeline_instance, tmp_path):
    video_path = tmp_path / "pan.mp4"
    # Pan velocity of 5 pixels per frame
    _create_synthetic_video(video_path, fps=30, duration_sec=2.0, pan_velocity_x=5.0, pan_velocity_y=-2.0)
    
    dx, dy = pipeline_instance._extract_camera_shift(video_path, 0.0, 1.0)
    
    # The camera panning right (+x direction) and up (-y direction)
    # The dx should be positive and dy should be negative.
    assert dx > 10.0
    assert dy < -5.0

def test_bystander_centered(pipeline_instance, tmp_path):
    video_path = tmp_path / "dummy.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=1.0)
    
    # Frame is 480x480. Middle 30% is X: [168, 312], Y: [168, 312]
    # Centered bystander
    bystanders = [{
        "timestamps_sec": [0.5],
        "bounding_boxes": [[200, 200, 280, 280]] # Center is 240, 240
    }]
    
    assert pipeline_instance._check_bystander_centering(video_path, bystanders, 0.0, 1.0) == True
    
    # Not centered bystander
    bystanders_not_centered = [{
        "timestamps_sec": [0.5],
        "bounding_boxes": [[10, 10, 50, 50]] # Center is 30, 30
    }]
    
    assert pipeline_instance._check_bystander_centering(video_path, bystanders_not_centered, 0.0, 1.0) == False

def test_full_pipeline_schema(pipeline_instance, tmp_path):
    video_path = tmp_path / "schema_test.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=2.0, pan_velocity_x=2.0)
    
    entry = {
        "video_id": "vid_03g_schema",
        "video_path": str(video_path),
        "bystander_detections": [
            {
                "person_id": 0,
                "timestamps_sec": [0.5],
                "bounding_boxes": [[200, 200, 280, 280]]
            }
        ],
        "identified_tasks": [
            {
                "task_id": "t_01",
                "task_temporal_metadata": {
                    "task_reaction_window_sec": [0.0, 1.0]
                }
            }
        ]
    }
    
    result = pipeline_instance.process_video(entry)
    
    assert result is not None
    assert result['video_id'] == "vid_03g_schema"
    assert result['layer'] == "03g_shared_reality"
    assert isinstance(result['tasks_analyzed'], list)
    for task in result['tasks_analyzed']:
        assert 'task_id' in task
        assert 'post_climax_camera_shift_vector' in task
        assert len(task['post_climax_camera_shift_vector']) == 2
        assert 'bystander_centered_in_fov' in task
        assert task['bystander_centered_in_fov'] is True
        assert 'social_reference_sought' in task
        assert isinstance(task['social_reference_sought'], bool)

def test_resumability_skips_processed(pipeline_instance, tmp_path):
    output_path = pipeline_instance.output_result_path

    existing = [{"video_id": "pre_existing", "layer": "03g_shared_reality", "tasks_analyzed": []}]
    with open(output_path, 'w') as f:
        json.dump(existing, f)

    pipeline2 = SharedRealityPipeline(
        pipeline_instance.input_manifest_path, output_path, force=False
    )
    assert "pre_existing" in pipeline2.processed_ids

def test_bystander_outside_reaction_window(pipeline_instance, tmp_path):
    """Bystander timestamps that fall entirely outside the reaction window
    should NOT trigger bystander_centered_in_fov, even if the centroid
    would otherwise land in the middle 30%."""
    video_path = tmp_path / "window_test.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=3.0)

    # Bystander at center (240,240) but at t=2.5 — outside the [0.0, 1.0] window
    bystanders = [{
        "timestamps_sec": [2.5],
        "bounding_boxes": [[200, 200, 280, 280]]
    }]

    result = pipeline_instance._check_bystander_centering(video_path, bystanders, 0.0, 1.0)
    assert result is False

def test_no_bystander_returns_none(pipeline_instance, tmp_path):
    """Videos with no bystander detections should return None."""
    video_path = tmp_path / "no_bystander.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=2.0)

    entry = {
        "video_id": "vid_no_bystander",
        "video_path": str(video_path),
        "bystander_detections": [],
        "identified_tasks": [
            {
                "task_id": "t_01",
                "task_temporal_metadata": {
                    "task_reaction_window_sec": [0.0, 1.0]
                }
            }
        ]
    }

    result = pipeline_instance.process_video(entry)
    assert result is None
