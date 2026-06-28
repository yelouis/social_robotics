"""
Tests for Layer 05 Visualizer.
"""
import pytest
from src.layer_05_visualizer.catalog import CatalogEntry, LAYER_FINDING_PREDICATES
from src.layer_05_visualizer.hydrate import build_overlay_bundle
from src.layer_05_visualizer.render import last_sample_at_or_before, nearest_sample

def test_findings_predicates():
    # 03a
    assert LAYER_FINDING_PREDICATES["03a"]({"aggregate": {"any_person_engaged": True}})
    assert not LAYER_FINDING_PREDICATES["03a"]({"aggregate": {"any_person_engaged": False}})
    
    # 03e
    assert LAYER_FINDING_PREDICATES["03e"]({"people": [{"gesture_detected": "affirming_nod"}]})
    assert not LAYER_FINDING_PREDICATES["03e"]({"people": [{"gesture_detected": "none"}]})

def test_bisect_helpers():
    samples = [{"t": 1.0, "val": "A"}, {"t": 3.0, "val": "B"}, {"t": 5.0, "val": "C"}]
    
    s = last_sample_at_or_before(samples, 0.5)
    assert s is None
    
    s = last_sample_at_or_before(samples, 1.0)
    assert s["val"] == "A"
    
    s = last_sample_at_or_before(samples, 4.0)
    assert s["val"] == "B"
    
    s = last_sample_at_or_before(samples, 6.0)
    assert s["val"] == "C"
    
    n = nearest_sample(samples, 1.8)
    assert n["val"] == "A"
    
    n = nearest_sample(samples, 2.2)
    assert n["val"] == "B"

def test_hydrate():
    entry = CatalogEntry(
        video_id="test-vid",
        manifest_entry={"video_width": 1920, "video_height": 1080, "fps": 30.0, "bystander_detections": [
            {"timestamp_sec": 1.0, "tracked_boxes": [{"person_id": 0, "box_2d": [0,0,192,108]}]}
        ]},
        results_by_layer={},
        video_path=None,
        findings={},
        num_layers_with_findings=0,
        summary_text="",
        sources={}
    )
    bundle = build_overlay_bundle(entry, probe_video=False)
    assert bundle["video_id"] == "test-vid"
    assert len(bundle["tracks"]) == 1
    
    # Check normalization
    box = bundle["tracks"][0]["boxes"][0]["box"]
    assert box[0] == 0.0
    assert box[1] == 0.0
    assert box[2] == 0.1 # 192/1920
    assert box[3] == 0.1 # 108/1080
