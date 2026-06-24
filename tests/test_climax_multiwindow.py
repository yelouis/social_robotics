"""Unit tests for the bystander-aware multi-window climax rework (02 Resolved #22).

Covers the pure helpers (clustering, bystander-timestamp extraction, the
iter/expand consumption helpers and legacy single-window back-compat) plus the
cluster->segment->schema shape of compute_task_climax_for_video driven by a fake
VideoCapture (no real decode). The optical-flow numerics themselves are covered
by the end-to-end run, so the fake frames here are uniform (flow = 0)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shared.climax_extraction import (  # noqa: E402
    _cluster_timestamps,
    _bystander_timestamps_in,
    iter_reaction_windows,
    expand_task_segments,
    compute_task_climax_for_video,
    CLIMAX_MAX_SEGMENTS,
)


# --- clustering ---
def test_cluster_splits_on_gap():
    cl = _cluster_timestamps([0, 1, 2, 30, 31, 32], gap_sec=15.0, max_span_sec=90.0)
    assert len(cl) == 2
    assert cl[0][2] == 3 and cl[1][2] == 3  # detection counts per cluster


def test_cluster_splits_on_max_span():
    ts = list(range(0, 200, 5))  # continuous for ~195 s; must split on the 90 s span cap
    cl = _cluster_timestamps(ts, gap_sec=15.0, max_span_sec=90.0)
    assert len(cl) >= 2
    assert all((c[1] - c[0]) <= 90.0 + 1e-6 for c in cl)


def test_cluster_empty():
    assert _cluster_timestamps([], 15.0, 90.0) == []


def test_bystander_timestamps_in_range():
    dets = [{"timestamps_sec": [1.0, 5.0, 100.0]}, {"timestamps_sec": [3.0, 50.0]}]
    assert sorted(_bystander_timestamps_in(dets, 0.0, 10.0)) == [1.0, 3.0, 5.0]
    assert _bystander_timestamps_in(None, 0.0, 10.0) == []


# --- consumption helpers ---
def _multi_task():
    return {"task_id": "t1", "task_velocity": "medium", "task_temporal_metadata": {"reaction_segments": [
        {"task_climax_sec": 10.0, "task_reaction_window_sec": [10.5, 12.0]},
        {"task_climax_sec": 50.0, "task_reaction_window_sec": [50.5, 52.0]},
    ]}}


def test_iter_reaction_windows_multi():
    assert list(iter_reaction_windows(_multi_task())) == [(10.0, [10.5, 12.0]), (50.0, [50.5, 52.0])]


def test_iter_reaction_windows_legacy_single():
    legacy = {"task_temporal_metadata": {"task_climax_sec": 5.0, "task_reaction_window_sec": [5.5, 7.0]}}
    assert list(iter_reaction_windows(legacy)) == [(5.0, [5.5, 7.0])]


def test_iter_reaction_windows_empty():
    assert list(iter_reaction_windows({"task_temporal_metadata": {"reaction_segments": []}})) == []
    assert list(iter_reaction_windows({})) == []


def test_expand_task_segments_multi():
    ps = list(expand_task_segments([_multi_task()]))
    assert len(ps) == 2
    assert [p["segment_index"] for p in ps] == [0, 1]
    assert all(p["n_segments"] == 2 for p in ps)
    assert all(p["task_id"] == "t1" for p in ps)  # original fields preserved
    for p, w in zip(ps, [[10.5, 12.0], [50.5, 52.0]]):
        # each pseudo-task is a clean single-window view
        assert p["task_temporal_metadata"]["task_reaction_window_sec"] == w
        assert "reaction_segments" not in p["task_temporal_metadata"]


def test_expand_legacy_and_empty():
    legacy = {"task_id": "t2", "task_temporal_metadata": {"task_climax_sec": 5.0, "task_reaction_window_sec": [5.5, 7.0]}}
    assert len(list(expand_task_segments([legacy]))) == 1
    no_window = {"task_id": "t3", "task_temporal_metadata": {"reaction_segments": []}}
    assert list(expand_task_segments([no_window])) == []  # nothing to process -> skipped


# --- compute_task_climax_for_video (fake decode) ---
class _FakeCap:
    """Returns uniform black frames; enough to exercise the cluster->segment path
    without a real video (optical flow is 0 on uniform frames)."""
    def set(self, *a):
        return True

    def grab(self):
        return True

    def read(self):
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        pass


def test_compute_emits_one_segment_per_cluster():
    fps, dur = 30.0, 300.0
    total = int(dur * fps)
    # two bystander clusters: ~10-19 s and ~200-209 s (gap 180 s > 15 s -> 2 clusters)
    dets = [{"timestamps_sec": list(np.arange(10, 20, 1.0))},
            {"timestamps_sec": list(np.arange(200, 210, 1.0))}]
    task = {"task_id": "t1", "task_start_sec": 0.0, "task_end_sec": dur,
            "task_velocity": "medium", "task_temporal_metadata": {}}
    compute_task_climax_for_video(_FakeCap(), fps, total, [task], dur,
                                  bystander_detections=dets, skip_vlm=True)
    meta = task["task_temporal_metadata"]
    segs = meta["reaction_segments"]
    assert len(segs) == 2                       # one segment per cluster
    assert meta["n_reaction_segments"] == 2
    assert "task_reaction_window_sec" in meta   # densest segment mirrored top-level
    # each segment's cluster lines up with the detections that produced it
    spans = sorted(s["bystander_cluster_sec"] for s in segs)
    assert spans[0][0] >= 9 and spans[0][1] <= 21
    assert spans[1][0] >= 199 and spans[1][1] <= 211


def test_compute_no_bystander_yields_no_segments():
    fps, dur = 30.0, 60.0
    task = {"task_id": "t1", "task_start_sec": 0.0, "task_end_sec": dur,
            "task_velocity": "medium", "task_temporal_metadata": {}}
    compute_task_climax_for_video(_FakeCap(), fps, int(dur * fps), [task], dur,
                                  bystander_detections=[], skip_vlm=True)
    meta = task["task_temporal_metadata"]
    assert meta["reaction_segments"] == []
    assert meta["climax_extraction_method"] == "no_bystander_cluster_in_task"
    assert list(iter_reaction_windows(task)) == []  # downstream correctly skips it


def test_compute_caps_segments():
    fps, dur = 30.0, 4000.0
    # 20 well-separated clusters -> capped at CLIMAX_MAX_SEGMENTS
    dets = [{"timestamps_sec": [float(c * 180 + k) for k in range(4)]} for c in range(20)]
    task = {"task_id": "t1", "task_start_sec": 0.0, "task_end_sec": dur,
            "task_velocity": "medium", "task_temporal_metadata": {}}
    compute_task_climax_for_video(_FakeCap(), fps, int(dur * fps), [task], dur,
                                  bystander_detections=dets, skip_vlm=True)
    assert len(task["task_temporal_metadata"]["reaction_segments"]) == CLIMAX_MAX_SEGMENTS
