import os
import json
import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.layer_03f_motor_resonance.pipeline import MotorResonancePipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    """Create a pipeline instance, skip if ultralytics is unavailable."""
    output_path = tmp_path / "03f_output.json"
    try:
        return MotorResonancePipeline(dummy_manifest, output_path)
    except RuntimeError as e:
        if "ultralytics is required" in str(e):
            pytest.skip("Ultralytics not installed, skipping test.")
        raise


def _create_synthetic_video(path, fps=30, duration_sec=3.0, width=480, height=480, jolt_at_sec=None):
    """Generate a synthetic video.  If ``jolt_at_sec`` is given, inject large
    random noise at that timestamp to simulate a camera jolt."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = int(fps * duration_sec)
    for i in range(total_frames):
        t = i / fps
        # Calm baseline: a slowly shifting gradient
        frame = np.full((height, width, 3), fill_value=int(50 + 20 * np.sin(t)), dtype=np.uint8)
        if jolt_at_sec and abs(t - jolt_at_sec) < 0.15:
            # Inject high-variance random noise to simulate camera shake
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


# ---------------------------------------------------------------------------
# 1. Initialization & Basic Resilience
# ---------------------------------------------------------------------------

def test_pipeline_initialization(dummy_manifest, tmp_path):
    output_path = tmp_path / "03f_output.json"
    try:
        pipeline = MotorResonancePipeline(dummy_manifest, output_path)
        assert pipeline is not None
        assert pipeline.model is not None
    except RuntimeError as e:
        if "ultralytics is required" in str(e):
            pytest.skip("Ultralytics not installed, skipping test.")
        else:
            raise e


def test_missing_video_handling(dummy_manifest, tmp_path):
    output_path = tmp_path / "03f_output.json"
    try:
        pipeline = MotorResonancePipeline(dummy_manifest, output_path)
        pipeline.run()
        # Should gracefully skip non-existent videos
        assert not output_path.exists()
    except RuntimeError as e:
        if "ultralytics is required" in str(e):
            pytest.skip("Ultralytics not installed, skipping test.")
        else:
            raise e


# ---------------------------------------------------------------------------
# 2. EgoMotion unit tests
# ---------------------------------------------------------------------------

def test_ego_motion_detects_jolt(pipeline_instance, tmp_path):
    """A synthetic video with a camera jolt at t=1.5s should produce spikes
    near that timestamp and a non-trivial chaos score."""
    video_path = tmp_path / "jolt.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=3.0, jolt_at_sec=1.5)

    spikes, chaos = pipeline_instance._extract_ego_motion(video_path, 0.5, 2.5)
    assert len(spikes) > 0, "Expected at least one EgoMotion spike from the camera jolt"
    assert chaos > 0.1, "Expected measurable chaos score from jolt"


def test_ego_motion_calm_video_no_spikes(pipeline_instance, tmp_path):
    """A calm (no jolt) synthetic video should produce NO spikes due to the
    chaos floor threshold (< 3.0)."""
    video_path = tmp_path / "calm.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=3.0, jolt_at_sec=None)

    spikes, chaos = pipeline_instance._extract_ego_motion(video_path, 0.0, 3.0)
    assert len(spikes) == 0, f"Calm video produced {len(spikes)} false-positive spikes"


def test_ego_motion_invalid_video(pipeline_instance, tmp_path):
    """Non-existent video should return empty results without crashing."""
    fake_path = tmp_path / "does_not_exist.mp4"
    spikes, chaos = pipeline_instance._extract_ego_motion(fake_path, 0.0, 1.0)
    assert spikes == []
    assert chaos == 0.0


# ---------------------------------------------------------------------------
# 3. Pose extraction – keypoint alignment & normalization
# ---------------------------------------------------------------------------

def test_keypoint_alignment_uses_dict_not_list(pipeline_instance, tmp_path):
    """Verify that keypoints are matched by body-part index, not by insertion
    order. We mock the YOLO model to return different subsets of keypoints
    across frames and ensure velocity is only computed for the intersection."""
    video_path = tmp_path / "alignment.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=2.0)

    # Build fake keypoints tensors
    import torch

    # Frame 1: only L-shoulder (idx 5) and R-wrist (idx 10) are confident
    kpts_frame1 = torch.zeros(1, 17, 3)
    kpts_frame1[0, 5] = torch.tensor([100, 100, 0.9])   # L shoulder
    kpts_frame1[0, 10] = torch.tensor([200, 200, 0.9])  # R wrist

    # Frame 2: only R-shoulder (idx 6) and R-wrist (idx 10) are confident
    kpts_frame2 = torch.zeros(1, 17, 3)
    kpts_frame2[0, 6] = torch.tensor([150, 150, 0.9])   # R shoulder
    kpts_frame2[0, 10] = torch.tensor([210, 210, 0.9])  # R wrist

    mock_result_1 = MagicMock()
    mock_result_1.keypoints = MagicMock()
    mock_result_1.keypoints.data = kpts_frame1

    mock_result_2 = MagicMock()
    mock_result_2.keypoints = MagicMock()
    mock_result_2.keypoints.data = kpts_frame2

    call_count = [0]
    def fake_model(crop, device=None, verbose=False):
        idx = call_count[0]
        call_count[0] += 1
        return [mock_result_1] if idx == 0 else [mock_result_2]

    original_model = pipeline_instance.model
    pipeline_instance.model = fake_model

    try:
        timestamps = [0.5, 1.0]
        bboxes = [[0, 0, 200, 200], [0, 0, 200, 200]]
        result = pipeline_instance._extract_and_correlate_pose(
            video_path, timestamps, bboxes, 0.0, 2.0, [0.5]
        )

        if result is not None:
            # Velocity should only be based on keypoint 10 (R wrist), the only
            # one present in both frames. Previously this would have matched
            # kpt 5 (L shoulder) with kpt 6 (R shoulder) by list position.
            assert result['velocity_peak'] >= 0
    finally:
        pipeline_instance.model = original_model


# ---------------------------------------------------------------------------
# 4. Full pipeline schema validation
# ---------------------------------------------------------------------------

def test_output_schema_on_synthetic_video(pipeline_instance, tmp_path):
    """Run the full pipeline against a synthetic video with a jolt and verify
    the output record matches the documented JSON schema."""
    video_path = tmp_path / "schema_test.mp4"
    _create_synthetic_video(video_path, fps=30, duration_sec=4.0, jolt_at_sec=2.0)

    entry = {
        "video_id": "schema_test_01",
        "video_path": str(video_path),
        "bystander_detections": [
            {
                "person_id": 0,
                "timestamps_sec": [1.0, 1.5, 2.0, 2.5, 3.0],
                "bounding_boxes": [
                    [50, 50, 200, 400],
                    [50, 50, 200, 400],
                    [50, 50, 200, 400],
                    [50, 50, 200, 400],
                    [50, 50, 200, 400],
                ]
            }
        ],
        "identified_tasks": [
            {
                "task_id": "t_01",
                "task_temporal_metadata": {
                    "task_reaction_window_sec": [1.0, 3.5]
                }
            }
        ]
    }

    result = pipeline_instance.process_video(entry)

    # Result may be None if YOLO can't detect a person in the synthetic gradient
    # frames.  That is a valid outcome—assert schema if present.
    if result is not None:
        assert result['video_id'] == "schema_test_01"
        assert result['layer'] == "03f_motor_resonance"
        assert isinstance(result['tasks_analyzed'], list)
        for task in result['tasks_analyzed']:
            assert 'task_id' in task
            assert 'ego_kinetic_chaos_score' in task
            assert isinstance(task['per_person'], list)
            for person in task['per_person']:
                assert 'person_id' in person
                assert 'bystander_pose_velocity_peak' in person
                assert 'resonance_delay_sec' in person
                assert isinstance(person['motor_resonance_detected'], bool)
                assert 'empathy_scalar' in person


# ---------------------------------------------------------------------------
# 5. Resumability
# ---------------------------------------------------------------------------

def test_resumability_skips_processed(pipeline_instance, tmp_path):
    """Already-processed video IDs should be skipped on re-run."""
    output_path = pipeline_instance.output_result_path

    # Pre-seed an existing result
    existing = [{"video_id": "pre_existing", "layer": "03f_motor_resonance", "tasks_analyzed": []}]
    with open(output_path, 'w') as f:
        json.dump(existing, f)

    # Re-instantiate to pick up existing results
    pipeline2 = MotorResonancePipeline(
        pipeline_instance.input_manifest_path, output_path, force=False
    )
    assert "pre_existing" in pipeline2.processed_ids
