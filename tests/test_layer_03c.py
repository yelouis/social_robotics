import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.layer_03c_acoustic_prosody.pipeline import AcousticProsodyPipeline

# ------------------------------------------------------------------
#  Fixtures
# ------------------------------------------------------------------

def _make_pipeline(manifest="dummy.json", output="dummy_out.json", force=False):
    """Create a pipeline instance with _init_models bypassed for unit tests.
    
    The real _init_models now raises RuntimeError if ffmpeg/librosa/funasr
    are missing, which is correct production behavior. For unit tests we
    bypass it and set flags manually so we can test the pure logic.
    """
    with patch.object(AcousticProsodyPipeline, "_init_models"):
        pipeline = AcousticProsodyPipeline(manifest, output, force)
    pipeline.model = None
    pipeline.funasr_available = False
    pipeline.librosa_available = False
    pipeline.ffmpeg_available = True
    return pipeline

@pytest.fixture
def dummy_manifest(tmp_path):
    manifest_path = tmp_path / "filtered_manifest.json"
    data = [
        {
            "video_id": "test_video_1",
            "video_path": "dummy.mp4",
            "identified_tasks": [
                {
                    "task_id": "t_01",
                    "task_label": "Yelling",
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
#  Startup validation tests
# ------------------------------------------------------------------

def test_init_raises_without_ffmpeg():
    """Pipeline raises RuntimeError with install instructions if ffmpeg is missing."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            AcousticProsodyPipeline("dummy.json", "dummy_out.json")

def test_init_raises_without_librosa():
    """Pipeline raises RuntimeError with install instructions if librosa is missing."""
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "librosa":
            raise ImportError("No module named 'librosa'")
        return real_import(name, *args, **kwargs)

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(RuntimeError, match="librosa is not installed"):
            AcousticProsodyPipeline("dummy.json", "dummy_out.json")

def test_init_raises_without_funasr():
    """Pipeline raises RuntimeError with install instructions if funasr is missing."""
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "funasr":
            raise ImportError("No module named 'funasr'")
        return real_import(name, *args, **kwargs)

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(RuntimeError, match="funasr is not installed"):
            AcousticProsodyPipeline("dummy.json", "dummy_out.json")

# ------------------------------------------------------------------
#  Heuristic classification tests
# ------------------------------------------------------------------

def test_pipeline_heuristics_alarming():
    """Verify that high angry/fearful + high volume classifies as Alarming (-1.0)."""
    pipeline = _make_pipeline()
    
    emotions = {
        "angry": 0.6,
        "fearful": 0.3,
        "neutral": 0.1
    }
    # -10 dbFS is high volume
    max_amp = -10.0
    pitch_var = 0.8
    
    tone, scalar = pipeline._classify_acoustic_tone(emotions, max_amp, pitch_var)
    
    assert tone == "Alarming"
    assert scalar == -1.0

def test_pipeline_heuristics_soothing():
    """Verify that high happy/surprised + melodic pitch classifies as Soothing (+1.0)."""
    pipeline = _make_pipeline()
    
    emotions = {
        "happy": 0.5,
        "surprised": 0.3,
        "neutral": 0.2
    }
    max_amp = -25.0
    # High pitch variance (melodic)
    pitch_var = 0.9 
    
    tone, scalar = pipeline._classify_acoustic_tone(emotions, max_amp, pitch_var)
    
    assert tone == "Soothing"
    assert scalar == 1.0

def test_pipeline_heuristics_discouraging():
    """Verify that high sad + low volume classifies as Discouraging (-0.5)."""
    pipeline = _make_pipeline()
    
    emotions = {
        "sad": 0.7,
        "neutral": 0.1,
        "angry": 0.05,
    }
    # -40 dbFS is low volume
    max_amp = -40.0
    pitch_var = 0.1
    
    tone, scalar = pipeline._classify_acoustic_tone(emotions, max_amp, pitch_var)
    
    assert tone == "Discouraging"
    assert scalar == -0.5

def test_pipeline_heuristics_neutral():
    """Verify that low confidence scores default to Neutral (0.0)."""
    pipeline = _make_pipeline()
    
    emotions = {
        "happy": 0.1,
        "sad": 0.1,
        "angry": 0.1,
        "neutral": 0.7
    }
    max_amp = -25.0
    pitch_var = 0.2 
    
    tone, scalar = pipeline._classify_acoustic_tone(emotions, max_amp, pitch_var)
    
    assert tone == "Neutral"
    assert scalar == 0.0

# ------------------------------------------------------------------
#  Audio extraction tests
# ------------------------------------------------------------------

@patch("src.layer_03c_acoustic_prosody.pipeline.subprocess.run")
@patch("src.layer_03c_acoustic_prosody.pipeline.tempfile.mkstemp")
def test_audio_chunk_extraction_mock(mock_mkstemp, mock_run):
    """Verify ffmpeg is called correctly and returns a valid path."""
    pipeline = _make_pipeline()

    fake_path = "/tmp/prosody_test12345.wav"
    mock_mkstemp.return_value = (99, fake_path)

    mock_run.return_value = MagicMock()

    with patch("os.close") as mock_close, \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1024):
        wav_path = pipeline._extract_audio_chunk("dummy.mp4", 1.0, 3.0)

    mock_close.assert_called_once_with(99)
    assert wav_path == fake_path
    assert wav_path.endswith(".wav")

def test_audio_chunk_invalid_window():
    """Verify that an invalid time window (start >= end) returns None."""
    pipeline = _make_pipeline()
    result = pipeline._extract_audio_chunk("dummy.mp4", 5.0, 3.0)
    assert result is None

@patch("src.layer_03c_acoustic_prosody.pipeline.subprocess.run")
@patch("src.layer_03c_acoustic_prosody.pipeline.tempfile.mkstemp")
def test_audio_chunk_header_only_guard(mock_mkstemp, mock_run):
    """Verify that a WAV file with only a header (<=44 bytes) is rejected."""
    pipeline = _make_pipeline()

    fake_path = "/tmp/prosody_empty.wav"
    mock_mkstemp.return_value = (99, fake_path)
    mock_run.return_value = MagicMock()

    with patch("os.close"), \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=44), \
         patch("os.remove"):
        wav_path = pipeline._extract_audio_chunk("dummy.mp4", 1.0, 3.0)

    assert wav_path is None

# ------------------------------------------------------------------
#  Resumability tests
# ------------------------------------------------------------------

def test_resumability(dummy_manifest, tmp_path):
    """Pipeline skips already-processed video IDs when --force is not set."""
    out_path = tmp_path / "03c_acoustic_prosody_result.json"
    dummy_result = [{"video_id": "test_video_1", "layer": "03c_acoustic_prosody", "tasks_analyzed": []}]
    with open(out_path, 'w') as f:
        json.dump(dummy_result, f)

    pipeline = _make_pipeline(str(dummy_manifest), str(out_path), force=False)
    pipeline.run()

    with open(out_path, 'r') as f:
        results = json.load(f)
    assert len(results) == 1
    assert results[0]["tasks_analyzed"] == []  # Not re-processed

# ------------------------------------------------------------------
#  Error logging tests
# ------------------------------------------------------------------

def test_error_logging(tmp_path):
    """Verify that _log_error writes to the dedicated errors JSON file."""
    out_path = tmp_path / "03c_acoustic_prosody_result.json"
    pipeline = _make_pipeline("dummy.json", str(out_path))

    try:
        raise ValueError("test error for logging")
    except ValueError as e:
        pipeline._log_error("video_abc", e)

    error_log = tmp_path / "03c_acoustic_prosody_errors.json"
    assert error_log.exists()
    with open(error_log, 'r') as f:
        errors = json.load(f)
    assert len(errors) == 1
    assert errors[0]["video_id"] == "video_abc"
    assert "test error for logging" in errors[0]["error"]

# ------------------------------------------------------------------
#  End-to-end schema conformance
# ------------------------------------------------------------------

def test_schema_conformance(dummy_manifest, tmp_path, monkeypatch):
    out_path = tmp_path / "03c_acoustic_prosody_result.json"
    pipeline = _make_pipeline(str(dummy_manifest), str(out_path))
    
    # Mock video exists
    monkeypatch.setattr("os.path.exists", lambda x: True)
    
    # Mock extract_audio
    monkeypatch.setattr(pipeline, "_extract_audio_chunk", lambda v, s, e: "dummy.wav")
    
    # Mock extract features
    monkeypatch.setattr(pipeline, "_extract_librosa_features", lambda w: (-12.4, 0.85))
    
    # Mock ser model
    monkeypatch.setattr(pipeline, "_run_ser_model", lambda w: {"angry": 0.72, "fearful": 0.1, "neutral": 0.18})
    
    # Mock file cleanup
    monkeypatch.setattr(AcousticProsodyPipeline, "_safe_remove", staticmethod(lambda x: None))
    
    pipeline.run()
    
    assert out_path.exists()
    
    with open(out_path, 'r') as f:
        results = json.load(f)
        
    assert len(results) == 1
    res = results[0]
    assert res['video_id'] == "test_video_1"
    assert res['layer'] == "03c_acoustic_prosody"
    assert "tasks_analyzed" in res
    assert len(res["tasks_analyzed"]) == 1
    
    task_res = res["tasks_analyzed"][0]
    assert task_res["task_id"] == "t_01"
    assert "prosody_metrics" in task_res
    
    metrics = task_res["prosody_metrics"]
    assert metrics["max_amplitude_dbFS"] == -12.4
    assert metrics["pitch_contour_variance"] == 0.85
    assert "emotion_scores" in metrics
    assert metrics["emotion_scores"]["angry"] == 0.72
    assert metrics["dominant_emotion"] == "angry"
    assert task_res["classified_acoustic_tone"] == "Alarming"
    assert task_res["prosody_scalar"] == -1.0
