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


def _pipe(man, **kw):
    kw.setdefault("face_verify", False)
    kw.setdefault("action_captions", False)
    return TaskClimaxPipeline(man, **kw)


def test_pipeline_annotates_and_is_idempotent(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps([_entry()]))
    n = _pipe(man).run()
    assert n == 1
    data = json.loads(man.read_text())
    meta = data[0]["identified_tasks"][0]["task_temporal_metadata"]
    assert meta["n_reaction_segments"] == 2       # two clusters (10-12 s, 200-201 s)
    # second run: cached, nothing to do
    assert _pipe(man).run() == 0


def test_pipeline_force_reannotates(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps([_entry()]))
    assert _pipe(man).run() == 1
    assert _pipe(man, force=True).run() == 1


def test_pipeline_entry_filter_skips(tmp_path):
    man = tmp_path / "m.json"
    man.write_text(json.dumps([_entry("keep"), _entry("skip")]))
    n = _pipe(man, entry_filter=lambda e: e["id"] == "keep").run()
    assert n == 1
    data = json.loads(man.read_text())
    by_id = {e["id"]: e for e in data}
    assert by_id["keep"]["identified_tasks"][0]["task_temporal_metadata"]
    assert not by_id["skip"]["identified_tasks"][0]["task_temporal_metadata"]


def test_pipeline_survives_missing_video(tmp_path):
    # video_path doesn't exist -> face verify silently degrades to manifest-only.
    man = tmp_path / "m.json"
    man.write_text(json.dumps([_entry()]))
    assert _pipe(man, face_verify=True).run() == 1
    meta = json.loads(man.read_text())[0]["identified_tasks"][0]["task_temporal_metadata"]
    assert meta["reaction_segments"][0]["climax_extraction_method"] == "bbox_kernel_peak"


# --- per-segment action captions (docs/06 Issue 1) ---
import numpy as np  # noqa: E402

import layer_02b_task_climax.pipeline as l02b  # noqa: E402
from layer_02b_task_climax.pipeline import compute_task_climax_for_video  # noqa: E402


class _FakeCap:
    def set(self, *a):
        return True

    def read(self):
        return True, np.zeros((48, 64, 3), dtype=np.uint8)


def _captionable_task():
    dets = [{"timestamps_sec": [10.0, 11.0, 12.0],
             "bounding_boxes": [[0, 0, 50, 200]] * 3}]
    task = {"task_id": "t1", "task_label": "Playing cards",
            "task_start_sec": 0.0, "task_end_sec": 60.0,
            "task_temporal_metadata": {}}
    return task, dets


def test_caption_recorded(monkeypatch):
    monkeypatch.setattr(l02b, "ollama_chat",
                        lambda *a, **k: "  hands a card to  the player opposite ")
    task, dets = _captionable_task()
    compute_task_climax_for_video(_FakeCap(), 30.0, 0, [task], 60.0,
                                  bystander_detections=dets, face_verify=False,
                                  vlm_model="qwen-test", skip_vlm=False)
    seg = task["task_temporal_metadata"]["reaction_segments"][0]
    assert seg["segment_action_caption"] == "hands a card to the player opposite"


def test_caption_sentinels_normalized(monkeypatch):
    monkeypatch.setattr(l02b, "ollama_chat", lambda *a, **k: "Unclear.")
    task, dets = _captionable_task()
    compute_task_climax_for_video(_FakeCap(), 30.0, 0, [task], 60.0,
                                  bystander_detections=dets, face_verify=False,
                                  vlm_model="qwen-test", skip_vlm=False)
    seg = task["task_temporal_metadata"]["reaction_segments"][0]
    assert seg["segment_action_caption"] == "unclear"


def test_caption_failure_is_isolated(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(l02b, "ollama_chat", boom)
    task, dets = _captionable_task()
    compute_task_climax_for_video(_FakeCap(), 30.0, 0, [task], 60.0,
                                  bystander_detections=dets, face_verify=False,
                                  vlm_model="qwen-test", skip_vlm=False)
    seg = task["task_temporal_metadata"]["reaction_segments"][0]
    assert "segment_action_caption" not in seg      # field absent, rest intact
    assert seg["climax_extraction_method"] == "bbox_kernel_peak"


def test_caption_off_on_lazy_path(monkeypatch):
    # skip_vlm=True (every pre-existing caller) must never call the VLM.
    called = []
    monkeypatch.setattr(l02b, "ollama_chat", lambda *a, **k: called.append(1) or "x")
    task, dets = _captionable_task()
    compute_task_climax_for_video(_FakeCap(), 30.0, 0, [task], 60.0,
                                  bystander_detections=dets, face_verify=False,
                                  vlm_model="qwen-test", skip_vlm=True)
    assert called == []
    assert "segment_action_caption" not in task["task_temporal_metadata"]["reaction_segments"][0]


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
