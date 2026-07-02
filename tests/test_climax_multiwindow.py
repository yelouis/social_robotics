"""Unit tests for the Layer 02b bbox-kernel climax detector (June 30 rework;
formerly the bystander-aware multi-window optical-flow detector, 02 Resolved #22).

Covers the pure helpers (clustering, detection extraction, kernel scoring, the
straddle window) plus the cluster->segment->schema shape of
compute_task_climax_for_video and the consumption helpers' back-compat.
All tests run with face_verify=False — the detector is manifest-only there
(no decode), so no fake VideoCapture is needed; the BlazeFace verification
path is exercised by the end-to-end run.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layer_02b_task_climax.pipeline import (  # noqa: E402
    _cluster_timestamps,
    _bystander_timestamps_in,
    _detections_in,
    _kernel_scores,
    _straddle_window,
    compute_task_climax_for_video,
    CLIMAX_MAX_SEGMENTS,
    WINDOW_PRE_SEC,
    WINDOW_POST_SEC,
)
from shared.climax_extraction import (  # noqa: E402  back-compat surface
    iter_reaction_windows,
    expand_task_segments,
    compute_task_climax_for_video as compute_via_shared,
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


# --- detection extraction + kernel scoring ---
def _track(ts, heights):
    return {"timestamps_sec": list(ts),
            "bounding_boxes": [[0, 0, 50, h] for h in heights]}


def test_detections_in_carries_bbox_height():
    dets = _detections_in([_track([1.0, 5.0], [100, 300])], 0.0, 10.0)
    assert [(t, h) for t, h, _ in dets] == [(1.0, 100.0), (5.0, 300.0)]


def test_detections_in_tolerates_missing_boxes():
    dets = _detections_in([{"timestamps_sec": [1.0, 2.0]}], 0.0, 10.0)
    assert [(t, h) for t, h, _ in dets] == [(1.0, 0.0), (2.0, 0.0)]


def test_kernel_prefers_sustained_tall_boxes():
    # a lone 400px spike at t=20 vs three 300px detections around t=10:
    # the kernel (sum over +/-2s neighbors) must prefer the sustained group.
    dets = _detections_in([_track([9.0, 10.0, 11.0, 20.0], [300, 300, 300, 400])], 0.0, 30.0)
    scored = _kernel_scores(dets)
    assert scored[0][1] == 10.0  # centre of the sustained group wins


def test_kernel_climax_is_a_detection_timestamp():
    dets = _detections_in([_track([3.0, 6.0, 9.0], [100, 200, 150])], 0.0, 30.0)
    scored = _kernel_scores(dets)
    assert scored[0][1] in (3.0, 6.0, 9.0)


# --- straddle window ---
def test_straddle_window_shape():
    assert _straddle_window(10.0, 100.0) == [10.0 - WINDOW_PRE_SEC, 10.0 + WINDOW_POST_SEC]


def test_straddle_window_shifts_at_clip_start():
    w = _straddle_window(0.2, 100.0)
    assert w[0] == 0.0 and (w[1] - w[0]) == WINDOW_PRE_SEC + WINDOW_POST_SEC


def test_straddle_window_shifts_at_clip_end_no_degenerate():
    # The flow-era clamp produced [duration, duration] here; the straddle
    # window must shift back instead, keeping its full span.
    dur = 50.0
    w = _straddle_window(49.9, dur)
    assert w[1] == dur
    assert (w[1] - w[0]) == WINDOW_PRE_SEC + WINDOW_POST_SEC


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


# --- compute_task_climax_for_video (manifest-only; face_verify off) ---
def test_compute_emits_one_segment_per_cluster():
    fps, dur = 30.0, 300.0
    # two bystander clusters: ~10-19 s and ~200-209 s (gap 180 s > 15 s -> 2 clusters)
    dets = [_track(np.arange(10, 20, 1.0), [200] * 10),
            _track(np.arange(200, 210, 1.0), [300] * 10)]
    task = {"task_id": "t1", "task_start_sec": 0.0, "task_end_sec": dur,
            "task_velocity": "medium", "task_temporal_metadata": {}}
    compute_task_climax_for_video(None, fps, 0, [task], dur,
                                  bystander_detections=dets, face_verify=False)
    meta = task["task_temporal_metadata"]
    segs = meta["reaction_segments"]
    assert len(segs) == 2                       # one segment per cluster
    assert meta["n_reaction_segments"] == 2
    assert "task_reaction_window_sec" in meta   # densest segment mirrored top-level
    # each segment's cluster lines up with the detections that produced it
    spans = sorted(s["bystander_cluster_sec"] for s in segs)
    assert spans[0][0] >= 9 and spans[0][1] <= 21
    assert spans[1][0] >= 199 and spans[1][1] <= 211
    for s in segs:
        # the climax is a detection timestamp inside its own cluster...
        cs, ce = s["bystander_cluster_sec"]
        assert cs <= s["task_climax_sec"] <= ce
        # ...and the window straddles it
        w = s["task_reaction_window_sec"]
        assert w[0] < s["task_climax_sec"] < w[1]
        assert s["climax_extraction_method"] == "bbox_kernel_peak"


def test_compute_no_bystander_yields_no_segments():
    fps, dur = 30.0, 60.0
    task = {"task_id": "t1", "task_start_sec": 0.0, "task_end_sec": dur,
            "task_velocity": "medium", "task_temporal_metadata": {}}
    compute_task_climax_for_video(None, fps, 0, [task], dur,
                                  bystander_detections=[], face_verify=False)
    meta = task["task_temporal_metadata"]
    assert meta["reaction_segments"] == []
    assert meta["climax_extraction_method"] == "no_bystander_cluster_in_task"
    assert list(iter_reaction_windows(task)) == []  # downstream correctly skips it


def test_compute_caps_segments():
    fps, dur = 30.0, 4000.0
    # 20 well-separated clusters -> capped at CLIMAX_MAX_SEGMENTS
    dets = [_track([float(c * 180 + k) for k in range(4)], [150] * 4) for c in range(20)]
    task = {"task_id": "t1", "task_start_sec": 0.0, "task_end_sec": dur,
            "task_velocity": "medium", "task_temporal_metadata": {}}
    compute_task_climax_for_video(None, fps, 0, [task], dur,
                                  bystander_detections=dets, face_verify=False)
    assert len(task["task_temporal_metadata"]["reaction_segments"]) == CLIMAX_MAX_SEGMENTS


def test_compute_skips_already_annotated():
    task = {"task_id": "t1", "task_start_sec": 0.0, "task_end_sec": 60.0,
            "task_temporal_metadata": {"reaction_segments": [{"task_climax_sec": 1.0,
                                                              "task_reaction_window_sec": [0.5, 3.0]}]}}
    before = task["task_temporal_metadata"]
    compute_task_climax_for_video(None, 30.0, 0, [task], 60.0,
                                  bystander_detections=[_track([1.0, 2.0], [100, 100])],
                                  face_verify=False)
    assert task["task_temporal_metadata"] is before  # untouched


def test_shared_wrapper_accepts_flow_era_kwargs():
    # tools/validate_climax.py & tests call with skip_vlm=...; the wrapper must
    # keep accepting (and ignoring) the retired flow-era kwargs.
    task = {"task_id": "t1", "task_start_sec": 0.0, "task_end_sec": 60.0,
            "task_velocity": "medium", "task_temporal_metadata": {}}
    compute_via_shared(None, 30.0, 0, [task], 60.0,
                       bystander_detections=[_track([1.0, 2.0, 3.0], [100, 100, 100])],
                       skip_vlm=True, vlm_model=None, face_verify=False)
    assert task["task_temporal_metadata"]["n_reaction_segments"] == 1
