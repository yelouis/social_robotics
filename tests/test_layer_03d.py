import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.layer_03d_proxemic_kinematics.pipeline import ProxemicKinematicsPipeline

# ------------------------------------------------------------------
#  Fixtures
# ------------------------------------------------------------------

def _make_pipeline(manifest="dummy.json", output="dummy_out.json", force=False):
    """Create a pipeline instance with mocked model initialization."""
    with patch.object(ProxemicKinematicsPipeline, "_init_model"):
        pipeline = ProxemicKinematicsPipeline(manifest, output, force)
    pipeline.depth_estimator = None
    return pipeline

@pytest.fixture
def dummy_manifest(tmp_path):
    manifest_path = tmp_path / "filtered_manifest.json"
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()
    
    data = [
        {
            "video_id": "test_video_1",
            "video_path": str(dummy_video),
            "bystander_detections": [
                {
                    "person_id": 0,
                    "timestamps_sec": [1.0, 1.5, 2.0, 2.5, 3.0],
                    "bounding_boxes": [
                        [100, 100, 200, 200],  # Area: 10000
                        [90, 90, 210, 210],    # Area: 14400
                        [80, 80, 220, 220],    # Area: 19600
                        [70, 70, 230, 230],    # Area: 25600
                        [60, 60, 240, 240]     # Area: 32400 (Approach, +224%)
                    ]
                }
            ],
            "identified_tasks": [
                {
                    "task_id": "t_01",
                    "task_label": "Handing Object",
                    "task_temporal_metadata": {
                        "task_reaction_window_sec": [1.0, 3.0]
                    }
                }
            ]
        }
    ]
    with open(manifest_path, 'w') as f:
        json.dump(data, f)
    return manifest_path

# ------------------------------------------------------------------
#  Heuristic classification tests
# ------------------------------------------------------------------

def test_bbox_scale_delta():
    """Verify that expanding bounding boxes yield positive scale delta."""
    pipeline = _make_pipeline()
    timestamps = [1.0, 2.0, 3.0]
    bboxes = [
        [100, 100, 200, 200], # 100x100 = 10000
        [50, 50, 250, 250],   # 200x200 = 40000
        [0, 0, 300, 300]      # 300x300 = 90000
    ]
    delta = pipeline._calculate_bbox_scale_delta(timestamps, bboxes, 1.0, 3.0)
    assert delta == 800.0  # (90000 - 10000) / 10000 * 100

def test_bbox_shrink_delta():
    """Verify that shrinking bounding boxes yield negative scale delta."""
    pipeline = _make_pipeline()
    timestamps = [1.0, 2.0, 3.0]
    bboxes = [
        [0, 0, 300, 300],      # 300x300 = 90000
        [100, 100, 200, 200]   # 100x100 = 10000
    ]
    # End frame is at 2.0, so area goes from 90000 to 10000
    delta = pipeline._calculate_bbox_scale_delta(timestamps, bboxes, 1.0, 2.0)
    assert round(delta, 2) == -88.89  # (10000 - 90000) / 90000 * 100

def test_proxemic_vector_approach():
    pipeline = _make_pipeline()
    # 25% increase in size -> norm_bbox = 0.5
    # depth_delta = -0.3 -> norm_depth = 0.6
    # vector = (0.5 * 0.4) + (0.6 * 0.6) = 0.2 + 0.36 = 0.56
    vector, action = pipeline._compute_proxemic_vector(25.0, -0.3)
    assert round(vector, 2) == 0.56
    assert action == "Approach_Intervention"

def test_proxemic_vector_avoidance():
    pipeline = _make_pipeline()
    # -25% decrease in size -> norm_bbox = -0.5
    # depth_delta = +0.2 -> norm_depth = -0.4
    # vector = (-0.5 * 0.4) + (-0.4 * 0.6) = -0.2 - 0.24 = -0.44
    vector, action = pipeline._compute_proxemic_vector(-25.0, 0.2)
    assert round(vector, 2) == -0.44
    assert action == "Avoidance"



# ------------------------------------------------------------------
#  End-to-end schema conformance
# ------------------------------------------------------------------

def test_schema_conformance(dummy_manifest, tmp_path, monkeypatch):
    out_path = tmp_path / "03d_proxemic_kinematics_result.json"
    pipeline = _make_pipeline(str(dummy_manifest), str(out_path))
    
    # Mock depth delta to simulate Depth Anything V2 returning a valid float
    monkeypatch.setattr(pipeline, "_calculate_depth_delta", lambda v, t, b, s, e: -0.25)
    
    # Run the pipeline
    pipeline.run()
    
    assert out_path.exists()
    
    with open(out_path, 'r') as f:
        results = json.load(f)
        
    assert len(results) == 1
    res = results[0]
    assert res['video_id'] == "test_video_1"
    assert res['layer'] == "03d_proxemic_kinematics"
    assert "tasks_analyzed" in res
    assert len(res["tasks_analyzed"]) == 1
    
    task_res = res["tasks_analyzed"][0]
    assert task_res["task_id"] == "t_01"
    assert "per_person" in task_res
    
    per_person = task_res["per_person"][0]
    assert per_person["person_id"] == 0
    assert "bbox_scale_delta_pct" in per_person
    assert per_person["bbox_scale_delta_pct"] == 224.0 # (32400-10000)/10000 * 100
    assert per_person["depth_anything_v2_delta"] == -0.25
    # vector = (1.0 * 0.4) + (0.5 * 0.6) = 0.4 + 0.3 = 0.7 > 0.3 -> Approach
    assert per_person["proxemic_vector"] == 0.7 
    assert per_person["classified_action"] == "Approach_Intervention"
    assert "proxemic_confidence" in per_person
    assert "optical_flow_noise" in per_person

def test_optical_flow_noise_rejection(dummy_manifest, tmp_path, monkeypatch):
    out_path = tmp_path / "03d_proxemic_kinematics_result_noise.json"
    pipeline = _make_pipeline(str(dummy_manifest), str(out_path))
    
    # Mock high noise > 15.0
    monkeypatch.setattr(pipeline, "_extract_ego_motion_noise", lambda v, s, e: 20.0)
    
    pipeline.run()
    
    assert out_path.exists()
    with open(out_path, 'r') as f:
        results = json.load(f)
        
    res = results[0]
    task_res = res["tasks_analyzed"][0]
    per_person = task_res["per_person"][0]
    
    assert per_person["optical_flow_noise"] == 20.0
    assert per_person["proxemic_vector"] == 0.0
    assert per_person["proxemic_confidence"] == 0.0
    assert per_person["classified_action"] == "Neutral"
