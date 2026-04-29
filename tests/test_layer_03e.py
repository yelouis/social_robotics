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

