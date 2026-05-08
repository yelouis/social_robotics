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

# ------------------------------------------------------------------
#  Issue 4: Extreme SSD mount validation (fail-fast in _init_model)
# ------------------------------------------------------------------

def test_init_raises_when_ssd_not_mounted(tmp_path):
    """_init_model must hard-fail when /Volumes/Extreme SSD is not actually
    a mount point — otherwise transformers silently spills 500MB of weights
    onto the boot disk under a phantom /Volumes/<name> directory.
    """
    pipeline = _make_pipeline()
    # Force the legitimate code path by clearing the bypass.
    with patch("src.layer_03d_proxemic_kinematics.pipeline.TRANSFORMERS_AVAILABLE", True), \
         patch("os.path.ismount", return_value=False):
        with pytest.raises(RuntimeError, match="Extreme SSD is not mounted"):
            pipeline._init_model()

# ------------------------------------------------------------------
#  Issue 6: tuning constants surface (subclass override)
# ------------------------------------------------------------------

def test_tuning_constants_subclass_override():
    """A subclass that lowers APPROACH_THRESHOLD reclassifies a borderline
    vector as Approach_Intervention without editing source — the documented
    rationale for hoisting the constants to a class-level block.
    """
    class SensitivePipeline(ProxemicKinematicsPipeline):
        APPROACH_THRESHOLD = 0.1

    with patch.object(SensitivePipeline, "_init_model"):
        pipeline = SensitivePipeline("dummy.json", "dummy_out.json")

    # bbox=10% (norm 0.2), depth=-0.05 (norm 0.1).
    # vector = 0.2*0.4 + 0.1*0.6 = 0.08 + 0.06 = 0.14 — Neutral under default
    # 0.3 threshold but Approach_Intervention under 0.1.
    vector, action = pipeline._compute_proxemic_vector(10.0, -0.05)
    assert round(vector, 2) == 0.14
    assert action == "Approach_Intervention"

# ------------------------------------------------------------------
#  Issue 2: linear regression slope-span delta
# ------------------------------------------------------------------

def test_slope_span_robust_to_endpoint_outlier():
    """The two-point endpoint delta would have read -0.5 (0.0 - 0.5). The
    slope-span fit through all samples weights the flat bulk at 0.3 and
    reports -0.4 — measurably attenuated by the intermediate samples that
    Issue 6's adaptive sampling exists to collect.
    """
    depths = [
        (1.0, 0.50),  # outlier on the high end
        (1.5, 0.30),
        (2.0, 0.30),
        (2.5, 0.30),
        (3.0, 0.00),  # outlier on the low end
    ]
    delta = ProxemicKinematicsPipeline._slope_span_delta(depths)
    endpoint_delta = depths[-1][1] - depths[0][1]  # -0.5
    assert round(delta, 4) == -0.4
    assert delta > endpoint_delta  # less negative -> outlier influence reduced

def test_slope_span_matches_endpoint_for_linear_data():
    """For a perfectly linear depth trajectory, the slope-span delta equals
    the simple last-minus-first delta — preserving the previous calibration
    constant in the absence of noise.
    """
    depths = [(t, 1.0 - 0.1 * t) for t in [1.0, 1.5, 2.0, 2.5, 3.0]]
    delta = ProxemicKinematicsPipeline._slope_span_delta(depths)
    # last - first = (1.0 - 0.3) - (1.0 - 0.1) = 0.7 - 0.9 = -0.2
    assert round(delta, 4) == -0.2

# ------------------------------------------------------------------
#  Issue 1: bbox-prompted SAM helper falls back gracefully
# ------------------------------------------------------------------

def test_segment_with_sam_returns_none_when_unloaded():
    """When SAM weights are not initialized, the helper must return None so
    `_calculate_depth_delta` falls back to the rectangular-bbox mask rather
    than crashing on attribute access.
    """
    pipeline = _make_pipeline()
    pipeline.sam_model = None
    pipeline.sam_processor = None
    img = MagicMock()
    assert pipeline._segment_with_sam(img, 0, 0, 100, 100) is None

# ------------------------------------------------------------------
#  Issue 3: sentinel record persists skip decision across resumes
# ------------------------------------------------------------------

def test_sentinel_record_for_missing_video(tmp_path):
    """When `process_video` returns None (e.g., missing video file), a
    sentinel record is appended and the video_id is marked processed so
    subsequent resume runs skip it instead of re-scanning optical flow.
    """
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([
        {
            "video_id": "missing_video",
            "video_path": str(tmp_path / "does_not_exist.mp4"),
            "bystander_detections": [],
            "identified_tasks": [],
        }
    ]))
    out_path = tmp_path / "out.json"
    pipeline = _make_pipeline(str(manifest_path), str(out_path))

    pipeline.run()

    with open(out_path, "r") as f:
        results = json.load(f)
    assert len(results) == 1
    assert results[0]["video_id"] == "missing_video"
    assert results[0]["tasks_analyzed"] == []
    assert results[0]["skipped_reason"] == "no_output_produced"
    assert "missing_video" in pipeline.processed_ids

# ------------------------------------------------------------------
#  Issue 5: per-video accelerator cache flush is safe on CPU
# ------------------------------------------------------------------

def test_release_accelerator_cache_noop_on_cpu():
    """The cache flush hook must be a no-op on CPU and survive without
    raising even if the torch backend lacks `mps`/`cuda` attributes.
    """
    pipeline = _make_pipeline()
    pipeline.device = "cpu"
    # Should not raise on CPU.
    pipeline._release_accelerator_cache()

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
