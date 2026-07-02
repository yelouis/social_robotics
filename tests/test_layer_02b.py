"""Layer 02b pipeline tests: manifest annotation contract (idempotent resume,
atomic write-back, entry_filter) plus the 03a segment-restrict helper it
enables. Detector numerics are covered by test_climax_multiwindow.py."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layer_02b_task_climax.pipeline import TaskClimaxPipeline  # noqa: E402
from layer_03a_attention.pipeline import _restrict_to_segments  # noqa: E402


def _entry(vid="v1", n_tasks=1):
    return {
        "id": vid,
        "video_path": f"/nonexistent/{vid}.mp4",   # face-verify skipped, manifest-only path
        "duration_sec": 300.0,
        "fps": 30.0,
        "bystander_detections": [{
            "person_id": 1,
            "timestamps_sec": [10.0, 11.0, 12.0, 200.0, 201.0],
            "bounding_boxes": [[0, 0, 50, 200]] * 5,
            "detection_confidence": [0.9] * 5,
        }],
        "identified_tasks": [
            {"task_id": f"t{k}", "task_start_sec": 0.0, "task_end_sec": 300.0,
             "task_velocity": "medium", "task_temporal_metadata": {}}
            for k in range(n_tasks)
        ],
    }


def test_pipeline_annotates_and_is_idempotent(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps([_entry()]))
    n = TaskClimaxPipeline(man, face_verify=False).run()
    assert n == 1
    data = json.loads(man.read_text())
    meta = data[0]["identified_tasks"][0]["task_temporal_metadata"]
    assert meta["n_reaction_segments"] == 2       # two clusters (10-12 s, 200-201 s)
    # second run: cached, nothing to do
    assert TaskClimaxPipeline(man, face_verify=False).run() == 0


def test_pipeline_force_reannotates(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps([_entry()]))
    assert TaskClimaxPipeline(man, face_verify=False).run() == 1
    assert TaskClimaxPipeline(man, face_verify=False, force=True).run() == 1


def test_pipeline_entry_filter_skips(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps([_entry("keep"), _entry("skip")]))
    n = TaskClimaxPipeline(man, face_verify=False,
                           entry_filter=lambda e: e["id"] == "keep").run()
    assert n == 1
    data = json.loads(man.read_text())
    by_id = {e["id"]: e for e in data}
    assert by_id["keep"]["identified_tasks"][0]["task_temporal_metadata"]
    assert not by_id["skip"]["identified_tasks"][0]["task_temporal_metadata"]


def test_pipeline_survives_missing_video(tmp_path):
    # video_path doesn't exist -> face verify silently degrades to manifest-only.
    man = tmp_path / "m.json"
    man.write_text(json.dumps([_entry()]))
    assert TaskClimaxPipeline(man, face_verify=True).run() == 1
    meta = json.loads(man.read_text())[0]["identified_tasks"][0]["task_temporal_metadata"]
    assert meta["reaction_segments"][0]["climax_extraction_method"] == "bbox_kernel_peak"


# --- 03a segment-restrict (C') ---
def _annotated_tasks():
    return [{"task_id": "t0", "task_temporal_metadata": {"reaction_segments": [
        {"task_climax_sec": 11.0, "task_reaction_window_sec": [10.0, 14.0]},
    ]}}]


def test_restrict_keeps_near_and_drops_far():
    tracks = [{"person_id": 1,
               "timestamps_sec": [11.0, 200.0],
               "bounding_boxes": [[0, 0, 10, 10], [0, 0, 20, 20]],
               "detection_confidence": [0.9, 0.8]}]
    out = _restrict_to_segments(tracks, _annotated_tasks(), margin_sec=35.0)
    assert len(out) == 1
    assert out[0]["timestamps_sec"] == [11.0]
    assert out[0]["bounding_boxes"] == [[0, 0, 10, 10]]   # co-indexed lists stay aligned
    assert out[0]["detection_confidence"] == [0.9]


def test_restrict_drops_empty_tracks():
    tracks = [{"person_id": 1, "timestamps_sec": [500.0], "bounding_boxes": [[0, 0, 1, 1]]}]
    assert _restrict_to_segments(tracks, _annotated_tasks(), margin_sec=35.0) == []


def test_restrict_fails_open_without_segments():
    tracks = [{"person_id": 1, "timestamps_sec": [500.0]}]
    # no annotated tasks -> unrestricted (Layer 02b not run yet)
    assert _restrict_to_segments(tracks, [{"task_id": "t0", "task_temporal_metadata": {}}]) is tracks
    assert _restrict_to_segments(tracks, None) is tracks
