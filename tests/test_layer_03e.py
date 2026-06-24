import json
import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.layer_03e_affirmation_gesture.pipeline import AffirmationGesturePipeline

@pytest.fixture
def mock_data_paths():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        manifest_path = temp_dir_path / "filtered_manifest.json"
        attention_path = temp_dir_path / "03a_attention_result.json"
        output_path = temp_dir_path / "03e_result.json"
        
        # 1. Create a dummy manifest
        manifest_data = [
            {
                "video_id": "vid_nod",
                "identified_tasks": [
                    {
                        "task_id": "task_1",
                        "task_temporal_metadata": {
                            "task_reaction_window_sec": [0.0, 5.0]
                        }
                    }
                ]
            },
            {
                "video_id": "vid_shake",
                "identified_tasks": [
                    {
                        "task_id": "task_2",
                        "task_temporal_metadata": {
                            "task_reaction_window_sec": [0.0, 5.0]
                        }
                    }
                ]
            },
            {
                "video_id": "vid_none",
                "identified_tasks": [
                    {
                        "task_id": "task_3",
                        "task_temporal_metadata": {
                            "task_reaction_window_sec": [0.0, 5.0]
                        }
                    }
                ]
            }
        ]
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f)
            
        # 2. Create dummy attention results
        # Time array: 0 to 5 seconds, 5 FPS
        t_arr = np.linspace(0, 5, 25)
        
        # Nod: Pitch oscillating at ~2Hz, Yaw flat
        pitch_nod = 0.1 * np.sin(2 * np.pi * 2.0 * t_arr)
        yaw_nod = 0.01 * np.random.randn(len(t_arr))
        
        trace_nod = [{"t": float(t), "score": 0.9, "pitch_rad": float(p), "yaw_rad": float(y)} 
                     for t, p, y in zip(t_arr, pitch_nod, yaw_nod)]
                     
        # Shake: Yaw oscillating at ~2Hz, Pitch flat
        pitch_shake = 0.01 * np.random.randn(len(t_arr))
        yaw_shake = 0.1 * np.sin(2 * np.pi * 2.0 * t_arr)
        
        trace_shake = [{"t": float(t), "score": 0.9, "pitch_rad": float(p), "yaw_rad": float(y)} 
                       for t, p, y in zip(t_arr, pitch_shake, yaw_shake)]
                       
        # None: Both relatively flat
        pitch_none = 0.01 * np.random.randn(len(t_arr))
        yaw_none = 0.01 * np.random.randn(len(t_arr))
        
        trace_none = [{"t": float(t), "score": 0.9, "pitch_rad": float(p), "yaw_rad": float(y)} 
                      for t, p, y in zip(t_arr, pitch_none, yaw_none)]
                      
        attention_data = [
            {
                "video_id": "vid_nod",
                "per_person": [{"person_id": 0, "attention_trace": trace_nod}]
            },
            {
                "video_id": "vid_shake",
                "per_person": [{"person_id": 0, "attention_trace": trace_shake}]
            },
            {
                "video_id": "vid_none",
                "per_person": [{"person_id": 0, "attention_trace": trace_none}]
            }
        ]
        
        with open(attention_path, 'w') as f:
            json.dump(attention_data, f)
            
        yield manifest_path, output_path, attention_path

def test_affirmation_gesture_logic(mock_data_paths):
    manifest_path, output_path, attention_path = mock_data_paths
    
    pipeline = AffirmationGesturePipeline(
        input_manifest_path=manifest_path,
        output_result_path=output_path,
        attention_result_path=attention_path,
        force=True
    )
    
    # 1. Test Nod
    entry_nod = {"video_id": "vid_nod", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    res_nod = pipeline.process_video(entry_nod)
    assert res_nod is not None
    person_res = res_nod['tasks_analyzed'][0]['per_person'][0]
    assert person_res['gesture_detected'] == 'affirming_nod'
    assert person_res['confidence'] > 0.6
    
    # 2. Test Shake
    entry_shake = {"video_id": "vid_shake", "identified_tasks": [{"task_id": "t2", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    res_shake = pipeline.process_video(entry_shake)
    assert res_shake is not None
    person_res = res_shake['tasks_analyzed'][0]['per_person'][0]
    assert person_res['gesture_detected'] == 'negating_shake'
    assert person_res['confidence'] > 0.6
    
    # 3. Test None
    entry_none = {"video_id": "vid_none", "identified_tasks": [{"task_id": "t3", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    res_none = pipeline.process_video(entry_none)
    assert res_none is not None
    person_res = res_none['tasks_analyzed'][0]['per_person'][0]
    assert person_res['gesture_detected'] == 'none'

def test_nan_resilience(mock_data_paths):
    manifest_path, output_path, attention_path = mock_data_paths
    
    pipeline = AffirmationGesturePipeline(
        input_manifest_path=manifest_path,
        output_result_path=output_path,
        attention_result_path=attention_path,
        force=True
    )
    
    # Test NaNs interpolation
    t_arr = np.linspace(0, 5, 25)
    pitch_nod = 0.1 * np.sin(2 * np.pi * 2.0 * t_arr)
    yaw_nod = 0.01 * np.random.randn(len(t_arr))
    
    # Introduce NaNs
    pitch_nod[5:10] = np.nan
    yaw_nod[12:15] = np.nan
    
    trace_nod = [{"t": float(t), "score": 0.9, "pitch_rad": float(p), "yaw_rad": float(y)} 
                 for t, p, y in zip(t_arr, pitch_nod, yaw_nod)]
                 
    pipeline.attention_data["vid_nan"] = {
        "video_id": "vid_nan",
        "per_person": [{"person_id": 0, "attention_trace": trace_nod}]
    }
    
    entry_nan = {"video_id": "vid_nan", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    res_nan = pipeline.process_video(entry_nan)
    
    assert res_nan is not None
    person_res = res_nan['tasks_analyzed'][0]['per_person'][0]
    # It should still detect the nod despite missing frames
    assert person_res['gesture_detected'] == 'affirming_nod'

def test_missing_attention_file_raises():
    """Layer 03e has a HARD dependency on 03a. Missing file must raise RuntimeError."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        manifest = [{"video_id": "v1"}]
        with open(td / "m.json", 'w') as f:
            json.dump(manifest, f)
        
        with pytest.raises(RuntimeError, match="HARD DEPENDENCY FAILURE"):
            AffirmationGesturePipeline(
                td / "m.json", td / "o.json", td / "NONEXISTENT.json", force=True
            )

def test_empty_attention_file_raises():
    """An empty 03a result file should also raise RuntimeError."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        manifest = [{"video_id": "v1"}]
        with open(td / "m.json", 'w') as f:
            json.dump(manifest, f)
        with open(td / "a.json", 'w') as f:
            json.dump([], f)  # empty list
        
        with pytest.raises(RuntimeError, match="is empty"):
            AffirmationGesturePipeline(
                td / "m.json", td / "o.json", td / "a.json", force=True
            )

def test_simultaneous_nod_shake_ambiguous(mock_data_paths):
    """When both pitch and yaw oscillate equally, the result should be ambiguous_wobble."""
    manifest_path, output_path, attention_path = mock_data_paths
    
    pipeline = AffirmationGesturePipeline(
        input_manifest_path=manifest_path,
        output_result_path=output_path,
        attention_result_path=attention_path,
        force=True
    )
    
    t_arr = np.linspace(0, 5, 25)
    # Both axes oscillating identically
    pitch_both = 0.1 * np.sin(2 * np.pi * 2.0 * t_arr)
    yaw_both = 0.1 * np.sin(2 * np.pi * 2.0 * t_arr)
    
    trace_both = [{"t": float(t), "score": 0.9, "pitch_rad": float(p), "yaw_rad": float(y)}
                  for t, p, y in zip(t_arr, pitch_both, yaw_both)]
    
    pipeline.attention_data["vid_both"] = {
        "video_id": "vid_both",
        "per_person": [{"person_id": 0, "attention_trace": trace_both}]
    }
    
    entry = {"video_id": "vid_both", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    res = pipeline.process_video(entry)
    assert res is not None
    person_res = res['tasks_analyzed'][0]['per_person'][0]
    assert person_res['gesture_detected'] == 'ambiguous_wobble'

def test_interpolation_threshold_calibrates_to_upstream_model():
    """Resolved Issue 16: MAX_INTERPOLATED_FRACTION re-keys against
    03a's processing_meta.model_used. Default 0.3 for unknown/L2CS, 0.2 for
    crossgaze, 0.25 for 3dgazenet."""
    cases = [
        ("l2cs_net_3d_gaze", 0.3),
        ("crossgaze", 0.2),
        ("3dgazenet", 0.25),
        ("future_unknown_model", 0.3),
        ("fallback_dummy", 0.3),
        (None, 0.3),
    ]
    for model, expected in cases:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            manifest = td / "m.json"
            attention = td / "a.json"
            output = td / "o.json"

            with open(manifest, 'w') as f:
                json.dump([], f)

            entry = {"video_id": "v1", "per_person": []}
            if model is not None:
                entry["processing_meta"] = {"model_used": model}
            with open(attention, 'w') as f:
                json.dump([entry], f)

            pipeline = AffirmationGesturePipeline(manifest, output, attention, force=True)
            assert pipeline.MAX_INTERPOLATED_FRACTION == expected, (model, pipeline.MAX_INTERPOLATED_FRACTION)


def test_nonuniform_sampling(mock_data_paths):
    """Verify the pipeline handles 03a's adaptive stride (non-uniform timestamps) correctly.
    
    Uses a 1Hz nod frequency which is within the Nyquist limit of the ~2.5Hz
    mean sampling rate. A 2Hz nod CANNOT be detected at 2.5Hz sampling (Nyquist violation).
    """
    manifest_path, output_path, attention_path = mock_data_paths
    
    pipeline = AffirmationGesturePipeline(
        input_manifest_path=manifest_path,
        output_result_path=output_path,
        attention_result_path=attention_path,
        force=True
    )
    
    # Build non-uniform timestamps simulating 03a's adaptive stride
    timestamps = [0.0]
    t = 0.0
    for i in range(24):
        if i % 3 == 0:
            t += 0.2  # fast sampling
        else:
            t += 0.5  # slow sampling
        timestamps.append(t)
    timestamps = np.array(timestamps)
    
    # 1Hz nod: detectable at mean fps ~2.5Hz (nyquist=1.25Hz > 1Hz)
    pitch = 0.15 * np.sin(2 * np.pi * 1.0 * timestamps)
    yaw = np.zeros_like(timestamps)
    
    trace = [{"t": float(t), "score": 0.9, "pitch_rad": float(p), "yaw_rad": float(y)}
             for t, p, y in zip(timestamps, pitch, yaw)]
    
    pipeline.attention_data["vid_nonuniform"] = {
        "video_id": "vid_nonuniform",
        "per_person": [{"person_id": 0, "attention_trace": trace}]
    }
    
    entry = {"video_id": "vid_nonuniform", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0.0, timestamps[-1]]}}]}
    res = pipeline.process_video(entry)
    assert res is not None
    person_res = res['tasks_analyzed'][0]['per_person'][0]
    assert person_res['gesture_detected'] == 'affirming_nod'


# ------------------------------------------------------------------
#  June 14 Issue 2: bystander-anchored read window
# ------------------------------------------------------------------

def _pipe(mock_data_paths):
    manifest_path, output_path, attention_path = mock_data_paths
    return AffirmationGesturePipeline(manifest_path, output_path, attention_path, force=True)


def test_measurement_window_keeps_dense_reaction_window(mock_data_paths):
    """>= MIN_TRACE_POINTS samples already inside the reaction window -> keep it
    (preserves the historical strict-window behavior; backward compatible)."""
    p = _pipe(mock_data_paths)
    trace_ts = [0.1 * i for i in range(40)]  # 0..3.9, dense
    win = p._measurement_window(trace_ts, [], climax_sec=2.5, start_sec=0.0, end_sec=5.0)
    assert win == (0.0, 5.0, "reaction_window")


def test_measurement_window_anchors_when_reaction_window_empty(mock_data_paths):
    """Issue 2: reaction window has no trace, but a bystander detection sits far
    from it near the climax -> anchor (padded) to that detection where 03a
    actually sampled."""
    p = _pipe(mock_data_paths)
    win = p._measurement_window([], [101.0], climax_sec=51.0, start_sec=50.0, end_sec=52.0)
    assert win == (101.0 - p.WINDOW_PAD_SEC, 101.0 + p.WINDOW_PAD_SEC, "bystander_anchored")


def test_measurement_window_span_capped(mock_data_paths):
    """Anchored span beyond MAX_ANCHOR_SPAN_SEC -> skip (span_capped)."""
    p = _pipe(mock_data_paths)
    win = p._measurement_window([], [0.0, 100.0], climax_sec=50.0, start_sec=49.0, end_sec=51.0)
    assert win[0] is None and win[2] == "span_capped"


def test_measurement_window_no_detection_to_anchor(mock_data_paths):
    """Empty reaction window AND no detections -> insufficient_trace."""
    p = _pipe(mock_data_paths)
    win = p._measurement_window([], [], climax_sec=5.0, start_sec=4.0, end_sec=6.0)
    assert win[0] is None and win[2] == "insufficient_trace"


def test_anchoring_recovers_offset_trace(mock_data_paths):
    """End-to-end Issue 2: trace lives at t=100-104 (around a bystander detection)
    while the reaction window is [50,52]. Pre-fix this scored insufficient_trace;
    anchoring must recover it into a real per-person result."""
    p = _pipe(mock_data_paths)
    ts = np.linspace(100.0, 104.0, 40)
    pitch = 0.15 * np.sin(2 * np.pi * 1.0 * ts)
    trace = [{"t": float(t), "score": 0.9, "pitch_rad": float(pp), "yaw_rad": 0.0}
             for t, pp in zip(ts, pitch)]
    p.attention_data["vid_off"] = {"video_id": "vid_off", "per_person": [{"person_id": 0, "attention_trace": trace}]}
    entry = {"video_id": "vid_off",
             "bystander_detections": [{"person_id": 0, "timestamps_sec": [102.0]}],
             "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {
                 "task_reaction_window_sec": [50.0, 52.0], "task_climax_sec": 51.0}}]}
    res = p.process_video(entry)
    assert "skipped_reason" not in res
    pr = res["tasks_analyzed"][0]["per_person"][0]
    assert pr["window_source"] == "bystander_anchored"


def test_dedup_and_untracked_filter(mock_data_paths):
    """Multi-window guardrails (Resolved #6): an UNTRACKED (negative-id) bystander
    is skipped, and a positive-id bystander whose multiple segments re-anchor to
    the SAME measurement window is scored once (not once per segment)."""
    p = _pipe(mock_data_paths)
    ts = np.linspace(100.0, 104.0, 40)
    pitch = 0.15 * np.sin(2 * np.pi * 1.0 * ts)
    trace = [{"t": float(t), "score": 0.9, "pitch_rad": float(pp), "yaw_rad": 0.0}
             for t, pp in zip(ts, pitch)]
    p.attention_data["vid_dd"] = {"video_id": "vid_dd", "per_person": [
        {"person_id": 0, "attention_trace": trace},     # genuine (tracked)
        {"person_id": -1, "attention_trace": trace},    # untracked -> must be skipped
    ]}
    # Two reaction segments at different climaxes; person 0's only detection is at
    # 102 s, so BOTH segments anchor to the same window -> must dedup to one record.
    entry = {"video_id": "vid_dd",
             "bystander_detections": [{"person_id": 0, "timestamps_sec": [102.0]},
                                      {"person_id": -1, "timestamps_sec": [102.0]}],
             "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {
                 "reaction_segments": [
                     {"task_climax_sec": 101.0, "task_reaction_window_sec": [50.0, 52.0]},
                     {"task_climax_sec": 103.0, "task_reaction_window_sec": [60.0, 62.0]},
                 ]}}]}
    res = p.process_video(entry)
    pids = [pp["person_id"] for t in res["tasks_analyzed"] for pp in t["per_person"]]
    assert -1 not in pids               # untracked fragment filtered
    assert pids.count(0) == 1           # genuine track scored once across the 2 collapsing segments


# ------------------------------------------------------------------
#  June 14 Issue 3: filtfilt padlen guard + per-person isolation
# ------------------------------------------------------------------

def test_safe_filtfilt_guards_short_signal():
    """filtfilt raises on len(sig) <= padlen(15); the guard returns None instead."""
    from scipy.signal import butter
    b, a = butter(2, [0.2, 0.8], btype='band')
    assert AffirmationGesturePipeline._safe_filtfilt(b, a, np.zeros(12)) is None
    out = AffirmationGesturePipeline._safe_filtfilt(b, a, np.sin(np.linspace(0, 10, 60)))
    assert out is not None and len(out) == 60


def test_short_window_does_not_crash(mock_data_paths):
    """Issue 3: a short window that resamples below filtfilt's padlen must fall
    back to peak-finding and return a record, not raise out and abort the clip."""
    p = _pipe(mock_data_paths)
    ts = np.linspace(0.0, 1.0, 10)  # ~9 fps -> nyq>3 -> bandpass branch, <16 samples
    pitch = 0.2 * np.sin(2 * np.pi * 1.5 * ts)
    trace = [{"t": float(t), "score": 0.9, "pitch_rad": float(pp), "yaw_rad": 0.0}
             for t, pp in zip(ts, pitch)]
    p.attention_data["vid_short"] = {"video_id": "vid_short", "per_person": [{"person_id": 0, "attention_trace": trace}]}
    entry = {"video_id": "vid_short",
             "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0.0, 1.0]}}]}
    res = p.process_video(entry)  # must not raise
    assert res["layer"] == "03e_affirmation_gesture"
    assert res["tasks_analyzed"][0]["per_person"][0]["person_id"] == 0


def test_per_person_failure_isolated(mock_data_paths, monkeypatch):
    """Issue 3: an unexpected per-person exception is contained — the clip emits a
    sentinel (skipped_reason='error') rather than propagating and losing the run."""
    p = _pipe(mock_data_paths)

    def boom(*a, **k):
        raise ValueError("synthetic detector failure")
    monkeypatch.setattr(p, "_detect_oscillation", boom)
    entry = {"video_id": "vid_nod",
             "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0.0, 5.0]}}]}
    res = p.process_video(entry)  # must not raise
    assert res["skipped_reason"] == "error"


# ------------------------------------------------------------------
#  June 14 Issue 2 (NoFace-zero): lost-tracking samples are missing data
# ------------------------------------------------------------------

def test_noface_samples_treated_as_missing(mock_data_paths):
    """`target='NoFace'` samples must become NaN (missing), not a real 0.0 reading,
    so a window dominated by NoFace is rejected (over_interpolated) instead of
    fabricating a step-edge 'nod'."""
    p = _pipe(mock_data_paths)
    t_arr = np.linspace(0, 5, 25)
    trace = []
    for i, t in enumerate(t_arr):
        if i < 14:  # 14/25 = 0.56 NoFace > MAX_INTERPOLATED_FRACTION (0.3)
            trace.append({"t": float(t), "score": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0, "target": "NoFace"})
        else:
            trace.append({"t": float(t), "score": 0.9, "pitch_rad": 0.01, "yaw_rad": 0.01, "target": "Face"})
    p.attention_data["vid_nf"] = {"video_id": "vid_nf", "per_person": [{"person_id": 0, "attention_trace": trace}]}
    entry = {"video_id": "vid_nf", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    res = p.process_video(entry)
    assert res["skipped_reason"] == "over_interpolated"


def test_small_noface_fraction_still_detects(mock_data_paths):
    """A few NoFace samples are bridged (and counted in interpolated_fraction) but a
    genuine nod still survives — NoFace is missing data, not a hard reject."""
    p = _pipe(mock_data_paths)
    t_arr = np.linspace(0, 5, 30)
    pitch = 0.1 * np.sin(2 * np.pi * 1.0 * t_arr)
    trace = []
    for i, (t, pp) in enumerate(zip(t_arr, pitch)):
        if i in (10, 11):  # 2/30 NoFace
            trace.append({"t": float(t), "score": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0, "target": "NoFace"})
        else:
            trace.append({"t": float(t), "score": 0.9, "pitch_rad": float(pp), "yaw_rad": 0.0, "target": "Face"})
    p.attention_data["vid_sm"] = {"video_id": "vid_sm", "per_person": [{"person_id": 0, "attention_trace": trace}]}
    entry = {"video_id": "vid_sm", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    res = p.process_video(entry)
    pr = res["tasks_analyzed"][0]["per_person"][0]
    assert pr["interpolated_fraction"] > 0  # the NoFace samples were counted as missing
    assert pr["gesture_detected"] == "affirming_nod"  # the real nod still detected


# ------------------------------------------------------------------
#  June 14 Issue 1: prefer true HEAD POSE over gaze (Option A)
# ------------------------------------------------------------------

def _hp_trace(t_arr, head_pitch=None, gaze_pitch=None):
    out = []
    for i, t in enumerate(t_arr):
        s = {"t": float(t), "score": 0.9, "target": "Face",
             "pitch_rad": float(gaze_pitch[i]) if gaze_pitch is not None else 0.0,
             "yaw_rad": 0.0}
        if head_pitch is not None:
            s["head_pitch_rad"] = head_pitch[i]
            s["head_yaw_rad"] = 0.0
        out.append(s)
    return out


def test_prefers_head_pose_and_ignores_gaze_oscillation(mock_data_paths):
    """The crux of Issue 1: oscillating GAZE with a FLAT head must NOT be a nod
    once head pose is available — 03e uses head pose and scores 'none'."""
    p = _pipe(mock_data_paths)
    t = np.linspace(0, 5, 30)
    trace = _hp_trace(t, head_pitch=[0.0] * len(t), gaze_pitch=0.15 * np.sin(2 * np.pi * 2.0 * t))
    p.attention_data["vid_hp"] = {"video_id": "vid_hp", "per_person": [{"person_id": 0, "attention_trace": trace}]}
    entry = {"video_id": "vid_hp", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    pr = p.process_video(entry)["tasks_analyzed"][0]["per_person"][0]
    assert pr["signal_source"] == "head_pose"
    assert pr["gesture_detected"] == "none"


def test_head_pose_detects_real_nod(mock_data_paths):
    """A genuine head-pitch oscillation is detected via head pose."""
    p = _pipe(mock_data_paths)
    t = np.linspace(0, 5, 30)
    trace = _hp_trace(t, head_pitch=list(0.15 * np.sin(2 * np.pi * 2.0 * t)))
    p.attention_data["vid_hn"] = {"video_id": "vid_hn", "per_person": [{"person_id": 0, "attention_trace": trace}]}
    entry = {"video_id": "vid_hn", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    pr = p.process_video(entry)["tasks_analyzed"][0]["per_person"][0]
    assert pr["signal_source"] == "head_pose"
    assert pr["gesture_detected"] == "affirming_nod"


def test_falls_back_to_gaze_without_head_fields(mock_data_paths):
    """Old 03a traces (no head_* fields) still work via gaze, unchanged."""
    p = _pipe(mock_data_paths)
    entry = {"video_id": "vid_nod", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    pr = p.process_video(entry)["tasks_analyzed"][0]["per_person"][0]
    assert pr["signal_source"] == "gaze"
    assert pr["gesture_detected"] == "affirming_nod"


def test_null_head_pose_is_missing_not_zero(mock_data_paths):
    """head_pitch_rad=None (FaceLandmarker tracking-loss) is treated as missing
    (NaN) — a window mostly without head pose is over_interpolated, not a nod."""
    p = _pipe(mock_data_paths)
    t = np.linspace(0, 5, 25)
    head_pitch = [None if i < 16 else 0.01 for i in range(len(t))]  # 16/25 missing
    trace = _hp_trace(t, head_pitch=head_pitch)
    p.attention_data["vid_hm"] = {"video_id": "vid_hm", "per_person": [{"person_id": 0, "attention_trace": trace}]}
    entry = {"video_id": "vid_hm", "identified_tasks": [{"task_id": "t1", "task_temporal_metadata": {"task_reaction_window_sec": [0, 5]}}]}
    assert p.process_video(entry)["skipped_reason"] == "over_interpolated"

