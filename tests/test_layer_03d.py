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
#  June 10 Issue 1: per-bystander window anchoring + detection-span widening
# ------------------------------------------------------------------

def test_measurement_window_keeps_reaction_window_when_dense():
    """>=2 detections inside the strict reaction window -> original window kept
    (the pre-existing dense-detection behavior is unchanged)."""
    pipeline = _make_pipeline()
    win = pipeline._bystander_measurement_window(
        [1.0, 1.5, 2.0, 2.5, 3.0], climax_sec=1.0, start_sec=1.0, end_sec=3.0)
    assert win == (1.0, 3.0, "reaction_window")


def test_measurement_window_anchors_sparse_detections():
    """The June 9 0/50 case: detections at the Node-02 ~6s cadence, a 2s
    reaction window holding one detection. The window must re-anchor to the
    climax-nearest detection +/- ANCHOR_SPAN_DETECTIONS (here [6, 18])."""
    pipeline = _make_pipeline()
    win = pipeline._bystander_measurement_window(
        [0.0, 6.0, 12.0, 18.0, 24.0], climax_sec=10.0, start_sec=10.0, end_sec=12.0)
    assert win == (6.0, 18.0, "bystander_anchored")


def test_measurement_window_anchor_clips_at_edges():
    """Anchor at the first/last detection still spans >=2 detections."""
    pipeline = _make_pipeline()
    win = pipeline._bystander_measurement_window(
        [0.0, 6.0, 12.0], climax_sec=0.0, start_sec=0.0, end_sec=2.0)
    assert win == (0.0, 6.0, "bystander_anchored")


def test_measurement_window_none_for_single_detection():
    """A bystander with a single detection can never yield a delta -> skipped
    with the 'single_detection' reason (Issue 2 sentinel provenance)."""
    pipeline = _make_pipeline()
    assert pipeline._bystander_measurement_window(
        [5.0], climax_sec=5.0, start_sec=4.0, end_sec=6.0) == (None, None, "single_detection")


def test_measurement_window_caps_long_anchored_span():
    """June 11 span cap (Resolved #3): a sparse track whose neighbor-detection
    gap stretches the anchored span past MAX_ANCHOR_SPAN_SEC measures
    locomotion/drift, not a reaction -> the bystander is skipped (None). This
    is the June 10 latent case (33/112 spans >30s, max 198s)."""
    pipeline = _make_pipeline()
    assert pipeline._bystander_measurement_window(
        [0.0, 100.0, 200.0], climax_sec=100.0, start_sec=99.0, end_sec=101.0) == (None, None, "span_capped")


def test_measurement_window_at_cap_boundary_kept():
    """A span exactly at MAX_ANCHOR_SPAN_SEC is still measurable (cap is
    strictly-greater-than)."""
    pipeline = _make_pipeline()
    win = pipeline._bystander_measurement_window(
        [0.0, 15.0, 30.0], climax_sec=15.0, start_sec=14.0, end_sec=16.0)
    assert win == (0.0, 30.0, "bystander_anchored")


def test_sparse_detections_now_score_end_to_end(tmp_path, monkeypatch):
    """June 10 Issue 1 end-to-end: the exact geometry that scored 0/50 on
    June 9 (6s detection cadence vs 2s window) now produces a scored record
    via the anchored window, with window provenance fields emitted."""
    manifest_path = tmp_path / "manifest.json"
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.touch()
    manifest_path.write_text(json.dumps([{
        "video_id": "sparse_video",
        "video_path": str(dummy_video),
        "bystander_detections": [{
            "person_id": 0,
            "timestamps_sec": [0.0, 6.0, 12.0, 18.0],
            "bounding_boxes": [
                [100, 100, 200, 200],   # 10000
                [90, 90, 210, 210],     # 14400
                [80, 80, 220, 220],     # 19600
                [70, 70, 230, 230],     # 25600
            ],
        }],
        "identified_tasks": [{
            "task_id": "t_01",
            "task_label": "Sparse task",
            "task_temporal_metadata": {
                "task_climax_sec": 11.0,
                "task_reaction_window_sec": [11.0, 13.0],  # holds only t=12
            },
        }],
    }]))
    out_path = tmp_path / "out.json"
    pipeline = _make_pipeline(str(manifest_path), str(out_path))
    monkeypatch.setattr(pipeline, "_extract_ego_motion_noise", lambda v, s, e: 2.0)
    monkeypatch.setattr(pipeline, "_calculate_depth_delta", lambda v, t, b, s, e: -0.1)

    pipeline.run()

    with open(out_path) as f:
        results = json.load(f)
    per_person = results[0]["tasks_analyzed"][0]["per_person"][0]
    # anchored to t=12 (nearest climax 11.0) +/-1 detection -> [6, 18]
    assert per_person["window_source"] == "bystander_anchored"
    assert per_person["measurement_window_sec"] == [6.0, 18.0]
    # bbox delta over [6,18]: (25600 - 14400) / 14400 * 100
    assert per_person["bbox_scale_delta_pct"] == round((25600 - 14400) / 14400 * 100, 2)
    assert per_person["classified_action"] in ("Approach_Intervention", "Neutral", "Avoidance")


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

# ------------------------------------------------------------------
#  June 13 Issue 1 Option C: identity-continuity guard
# ------------------------------------------------------------------

def test_identity_continuity_flags_collision():
    """The real 599f2f09 collision: person_id reused across a banner-holder
    (large, center) and a distant child (small, left). Consecutive in-window
    boxes have ~0 IoU AND a large area drop -> flagged discontinuous."""
    pipeline = _make_pipeline()
    min_iou, disc = pipeline._window_identity_continuity(
        [9.0, 21.0], [[579, 266, 781, 660], [485, 617, 548, 859]], 9.0, 21.0)
    assert disc is True
    assert min_iou < pipeline.IDENTITY_IOU_FLOOR


def test_identity_continuity_spares_camera_pan():
    """A same-area box that jumps across the frame (IoU 0, area ratio ~1.0) is
    one person displaced by a camera pan, NOT an identity switch -> not flagged
    (this is what spares real approach/recede)."""
    pipeline = _make_pipeline()
    _, disc = pipeline._window_identity_continuity(
        [1.0, 2.0], [[100, 100, 200, 300], [400, 100, 500, 300]], 1.0, 2.0)
    assert disc is False


def test_identity_continuity_spares_smooth_track():
    """A continuously-tracked person's overlapping boxes stay well above the
    IoU floor -> not flagged (the genuine 343f4d2d Approach behaves this way)."""
    pipeline = _make_pipeline()
    min_iou, disc = pipeline._window_identity_continuity(
        [1.0, 2.0, 3.0], [[100, 100, 200, 200], [95, 95, 205, 205], [90, 90, 210, 210]], 1.0, 3.0)
    assert disc is False
    assert min_iou > 0.3


def test_identity_continuity_single_box_is_safe():
    """<2 in-window detections -> no pair to compare -> (1.0, False)."""
    pipeline = _make_pipeline()
    assert pipeline._window_identity_continuity([1.0], [[0, 0, 10, 10]], 1.0, 3.0) == (1.0, False)


def test_identity_discontinuity_zeroes_vector_end_to_end(tmp_path, monkeypatch):
    """End-to-end: a chaos-surviving colliding person_id is zeroed with
    identity_discontinuity provenance, and the expensive depth pass never runs
    (the guard sits after the chaos gate but before depth)."""
    def _boom(*a, **k):
        raise AssertionError("depth must not run for an identity-rejected vector")
    manifest_path = tmp_path / "m.json"
    v = tmp_path / "v.mp4"
    v.touch()
    manifest_path.write_text(json.dumps([{
        "video_id": "collide",
        "video_path": str(v),
        "bystander_detections": [{
            "person_id": 2,
            "timestamps_sec": [9.0, 21.0],
            "bounding_boxes": [[579, 266, 781, 660], [485, 617, 548, 859]],
        }],
        "identified_tasks": [{
            "task_id": "t_01",
            "task_temporal_metadata": {"task_climax_sec": 15.0, "task_reaction_window_sec": [9.0, 21.0]},
        }],
    }]))
    out = tmp_path / "o.json"
    pipeline = _make_pipeline(str(manifest_path), str(out))
    # Low ego-motion -> survives the chaos gate, so the identity guard is what
    # rejects it; depth must still never run.
    monkeypatch.setattr(pipeline, "_extract_ego_motion_noise", lambda v, s, e: 2.0)
    monkeypatch.setattr(pipeline, "_calculate_depth_delta", _boom)
    pipeline.run()
    rec = json.load(open(out))[0]["tasks_analyzed"][0]["per_person"][0]
    assert rec["identity_discontinuity"] is True
    assert rec["proxemic_vector"] == 0.0
    assert rec["classified_action"] == "Neutral"
    assert rec["optical_flow_noise"] == 2.0
    assert rec["min_consecutive_iou"] < 0.1


def test_min_consecutive_iou_emitted_on_scored_record(dummy_manifest, tmp_path, monkeypatch):
    """Every scored vector carries min_consecutive_iou provenance, and a smooth
    track is not flagged as a discontinuity."""
    out = tmp_path / "o.json"
    pipeline = _make_pipeline(str(dummy_manifest), str(out))
    monkeypatch.setattr(pipeline, "_extract_ego_motion_noise", lambda v, s, e: 2.0)
    monkeypatch.setattr(pipeline, "_calculate_depth_delta", lambda v, t, b, s, e: -0.25)
    pipeline.run()
    rec = json.load(open(out))[0]["tasks_analyzed"][0]["per_person"][0]
    assert "min_consecutive_iou" in rec
    assert rec["min_consecutive_iou"] > 0.3
    assert rec.get("identity_discontinuity", False) is False

# ------------------------------------------------------------------
#  June 13 Issue 2: provenance-only sentinel reasons
# ------------------------------------------------------------------

def _sentinel_manifest(tmp_path, bystanders, tasks=None):
    v = tmp_path / "v.mp4"
    v.touch()
    mp_ = tmp_path / "m.json"
    mp_.write_text(json.dumps([{
        "video_id": "sent",
        "video_path": str(v),
        "bystander_detections": bystanders,
        "identified_tasks": tasks if tasks is not None else [{
            "task_id": "t_01",
            "task_temporal_metadata": {"task_climax_sec": 100.0, "task_reaction_window_sec": [99.0, 101.0]},
        }],
    }]))
    return mp_


def test_sentinel_reason_all_span_capped(tmp_path):
    mp_ = _sentinel_manifest(tmp_path, [{
        "person_id": 0, "timestamps_sec": [0.0, 100.0, 200.0],
        "bounding_boxes": [[0, 0, 10, 10]] * 3,
    }])
    out = tmp_path / "o.json"
    p = _make_pipeline(str(mp_), str(out))
    p.run()
    rec = json.load(open(out))[0]
    assert rec["tasks_analyzed"] == []
    assert rec["skipped_reason"] == "all_bystanders_span_capped"


def test_sentinel_reason_all_single_detection(tmp_path):
    mp_ = _sentinel_manifest(tmp_path, [{
        "person_id": 0, "timestamps_sec": [5.0], "bounding_boxes": [[0, 0, 10, 10]],
    }], tasks=[{"task_id": "t_01", "task_temporal_metadata": {
        "task_climax_sec": 5.0, "task_reaction_window_sec": [4.0, 6.0]}}])
    out = tmp_path / "o.json"
    p = _make_pipeline(str(mp_), str(out))
    p.run()
    assert json.load(open(out))[0]["skipped_reason"] == "all_bystanders_single_detection"


def test_sentinel_reason_mixed_skip(tmp_path):
    mp_ = _sentinel_manifest(tmp_path, [
        {"person_id": 0, "timestamps_sec": [0.0, 100.0, 200.0], "bounding_boxes": [[0, 0, 10, 10]] * 3},
        {"person_id": 1, "timestamps_sec": [100.0], "bounding_boxes": [[0, 0, 10, 10]]},
    ])
    out = tmp_path / "o.json"
    p = _make_pipeline(str(mp_), str(out))
    p.run()
    assert json.load(open(out))[0]["skipped_reason"] == "mixed_skip"


def test_sentinel_reason_no_bystanders(tmp_path):
    mp_ = _sentinel_manifest(tmp_path, [])
    out = tmp_path / "o.json"
    p = _make_pipeline(str(mp_), str(out))
    p.run()
    assert json.load(open(out))[0]["skipped_reason"] == "no_bystanders"
