"""Extensive tests for the shared bystander measurement-window helper.

Covers the helper directly across every branch + parameter, and pins the exact
03d and 03e configurations so a future refactor of either layer's wrapper is
caught here (in addition to the layers' own end-to-end suites)."""
import pytest

from src.shared.bystander_window import bystander_measurement_window as bw


# ---- 03d configuration: detections counted, no pad, >=2 required ----
D = dict(min_in_reaction_window=2, anchor_span_detections=1, pad_sec=0.0,
         max_anchor_span_sec=30.0, allow_single_detection=False,
         no_detection_reason="single_detection")
# ---- 03e configuration: trace counted, +/-2s pad, single ok ----
E = dict(min_in_reaction_window=5, anchor_span_detections=1, pad_sec=2.0,
         max_anchor_span_sec=30.0, allow_single_detection=True,
         no_detection_reason="insufficient_trace")


# ------------------------------------------------------------------
#  Core branches
# ------------------------------------------------------------------

def test_keeps_reaction_window_when_dense():
    assert bw([1.0, 1.5, 2.0, 2.5, 3.0], 1.0, 1.0, 3.0, **D) == (1.0, 3.0, "reaction_window")


def test_anchors_when_window_sparse():
    # 0 detections in [10,12]; climax-nearest is 12 (idx2); +/-1 -> ts[1]..ts[3]
    assert bw([0.0, 6.0, 12.0, 18.0, 24.0], 10.0, 10.0, 12.0, **D) == (6.0, 18.0, "bystander_anchored")


def test_anchor_clips_at_first_detection():
    assert bw([0.0, 6.0, 12.0], 0.0, 0.0, 2.0, **D) == (0.0, 6.0, "bystander_anchored")


def test_anchor_clips_at_last_detection():
    # climax at the end -> nearest is last idx; lo clips to idx-1, hi to last
    assert bw([0.0, 6.0, 12.0], 12.0, 11.0, 13.0, **D) == (6.0, 12.0, "bystander_anchored")


# ------------------------------------------------------------------
#  Single / no detection
# ------------------------------------------------------------------

def test_single_detection_disallowed_03d():
    assert bw([5.0], 5.0, 4.0, 6.0, **D) == (None, None, "single_detection")


def test_no_detections_uses_reason_03d():
    assert bw([], 5.0, 4.0, 6.0, **D) == (None, None, "single_detection")


def test_no_detections_uses_reason_03e():
    assert bw([], 5.0, 4.0, 6.0, **E) == (None, None, "insufficient_trace")


def test_single_detection_allowed_padded_03e():
    # no trace in window, one detection at 101 -> padded +/-2s
    assert bw([101.0], 51.0, 50.0, 52.0, keep_timestamps=[], **E) == (99.0, 103.0, "bystander_anchored")


def test_duplicate_timestamps_no_pad_degenerate():
    # two identical detections, no pad -> zero-width span -> single_detection
    assert bw([5.0, 5.0], 5.0, 100.0, 102.0, **D) == (None, None, "single_detection")


def test_duplicate_timestamps_padded_ok():
    # same duplicates but padded -> a real interval
    assert bw([5.0, 5.0], 5.0, 100.0, 102.0, keep_timestamps=[], **E) == (3.0, 7.0, "bystander_anchored")


# ------------------------------------------------------------------
#  Span cap
# ------------------------------------------------------------------

def test_span_cap_rejects_long_anchor():
    assert bw([0.0, 100.0, 200.0], 100.0, 99.0, 101.0, **D) == (None, None, "span_capped")


def test_span_cap_boundary_inclusive():
    # exactly 30s span is kept (cap is strictly-greater-than)
    assert bw([0.0, 15.0, 30.0], 15.0, 14.0, 16.0, **D) == (0.0, 30.0, "bystander_anchored")


def test_span_cap_counts_padding():
    # 03e: detections 0..28 (28s) + 2*2s pad = 32s > 30 -> capped
    assert bw([0.0, 28.0], 14.0, 100.0, 102.0, keep_timestamps=[], **E) == (None, None, "span_capped")


# ------------------------------------------------------------------
#  keep_timestamps (03e: trace counted separately from anchor detections)
# ------------------------------------------------------------------

def test_keep_timestamps_decouples_dense_test():
    # detections sparse (would anchor), but the TRACE is dense in-window -> keep
    trace = [0.1 * i for i in range(60)]  # 0..5.9
    assert bw([0.0, 30.0], 2.5, 0.0, 5.0, keep_timestamps=trace, **E) == (0.0, 5.0, "reaction_window")


def test_keep_defaults_to_detections_when_omitted():
    # 03d-style: no keep_timestamps -> detections are counted for the dense test
    assert bw([1.0, 2.0], 1.5, 1.0, 3.0, **D) == (1.0, 3.0, "reaction_window")


# ------------------------------------------------------------------
#  Tie-breaking + ordering
# ------------------------------------------------------------------

def test_climax_equidistant_tie_picks_lower_index():
    # climax 50 equidistant from 0 and 100 -> min() picks first (idx0)
    assert bw([0.0, 100.0], 50.0, 49.0, 51.0, anchor_span_detections=1, pad_sec=0.0,
              min_in_reaction_window=2, max_anchor_span_sec=1000.0,
              allow_single_detection=False, no_detection_reason="x") == (0.0, 100.0, "bystander_anchored")


def test_unsorted_detections_are_sorted():
    assert bw([18.0, 0.0, 12.0, 24.0, 6.0], 10.0, 10.0, 12.0, **D) == (6.0, 18.0, "bystander_anchored")


def test_wider_anchor_span():
    # anchor_span_detections=2 widens to +/-2 indices
    cfg = dict(D); cfg["anchor_span_detections"] = 2; cfg["max_anchor_span_sec"] = 1000.0
    assert bw([0.0, 6.0, 12.0, 18.0, 24.0], 12.0, 100.0, 102.0, **cfg) == (0.0, 24.0, "bystander_anchored")
