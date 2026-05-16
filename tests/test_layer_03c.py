import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.layer_03c_acoustic_prosody.pipeline import (
    AcousticProsodyPipeline,
    SENSEVOICE_EVENT_TOKENS,
)
from src.layer_03c_acoustic_prosody.config import Layer03cConfig

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

def test_init_raises_when_automodel_load_fails():
    """Issue #1: non-ImportError AutoModel failures (corrupt weights, OOM,
    ModelScope outage, etc.) must hard-fail rather than silently degrade to
    all-Neutral output. The original except clause merely logged these,
    re-introducing the symptom Issue #6 was supposed to eliminate.
    """
    fake_funasr = MagicMock()
    fake_funasr.AutoModel = MagicMock(side_effect=OSError("corrupt model weights"))

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict("sys.modules", {"funasr": fake_funasr}):
        with pytest.raises(RuntimeError, match="Failed to load funasr SER model"):
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
#  SenseVoice event detection (bracketed-only, lazy load)
# ------------------------------------------------------------------

def test_sensevoice_only_matches_bracketed_tokens():
    """Issue #3: bare-word substring matching produced false positives on
    transcribed speech. Only bracketed event tokens should now register.
    """
    pipeline = _make_pipeline()
    pipeline.funasr_available = True
    pipeline.sensevoice_model = MagicMock()
    # Transcribed speech containing the substring "laughter" inside other
    # words ("slaughter") and the bare word; neither should match.
    pipeline.sensevoice_model.generate.return_value = [
        {"text": "the slaughter scene was harrowing and laughter died down"}
    ]

    events = pipeline._run_sensevoice_model("dummy.wav")
    assert events == []

def test_sensevoice_matches_bracketed_token_emission():
    """Bracketed tokens emitted by SenseVoice are detected correctly."""
    pipeline = _make_pipeline()
    pipeline.funasr_available = True
    pipeline.sensevoice_model = MagicMock()
    pipeline.sensevoice_model.generate.return_value = [
        {"text": "<|laughter|> some speech <|applause|>"}
    ]

    events = pipeline._run_sensevoice_model("dummy.wav")
    assert sorted(events) == ["applause", "laughter"]

def test_sensevoice_lazy_loaded_on_first_low_confidence():
    """Issue #4: SenseVoice is constructed only on first invocation, not at init."""
    pipeline = _make_pipeline()
    pipeline.funasr_available = True
    pipeline.sensevoice_model = None  # baseline: not yet loaded
    fake_loaded_model = MagicMock()
    fake_loaded_model.generate.return_value = [{"text": "<|cough|>"}]
    pipeline._AutoModel = MagicMock(return_value=fake_loaded_model)

    events = pipeline._run_sensevoice_model("dummy.wav")

    pipeline._AutoModel.assert_called_once_with(
        model="iic/SenseVoiceSmall", disable_update=True
    )
    assert pipeline.sensevoice_model is fake_loaded_model
    assert events == ["cough"]

    # Second call must reuse the loaded model (no re-construction).
    pipeline._run_sensevoice_model("dummy.wav")
    pipeline._AutoModel.assert_called_once()

# ------------------------------------------------------------------
#  SenseVoice eager-load host gate (Issue #1)
# ------------------------------------------------------------------

def test_host_can_eager_load_sensevoice_gate():
    """Issue #1: the 48 GB unified-memory gate — hosts at or above the
    threshold (e.g. the 64 GB Mac Studio M4 Max) eager-load SenseVoice,
    hosts below it (e.g. the 24 GB Mac mini M4 Pro) stay lazy.
    """
    fake_vm = MagicMock()

    fake_vm.total = 64 * 2**30
    with patch("psutil.virtual_memory", return_value=fake_vm):
        assert AcousticProsodyPipeline.host_can_eager_load_sensevoice() is True

    fake_vm.total = 48 * 2**30
    with patch("psutil.virtual_memory", return_value=fake_vm):
        assert AcousticProsodyPipeline.host_can_eager_load_sensevoice() is True

    fake_vm.total = 24 * 2**30
    with patch("psutil.virtual_memory", return_value=fake_vm):
        assert AcousticProsodyPipeline.host_can_eager_load_sensevoice() is False

def test_sensevoice_eager_loaded_on_high_memory_host():
    """Issue #1: on hosts with >= 48 GB unified memory, SenseVoice is
    eager-loaded during _init_models so the low-confidence gate never pays
    the ~2-3s load latency mid-run.
    """
    fake_funasr = MagicMock()
    emotion_model = MagicMock(name="emotion2vec")
    sensevoice_model = MagicMock(name="sensevoice")
    fake_funasr.AutoModel = MagicMock(side_effect=[emotion_model, sensevoice_model])

    fake_vm = MagicMock()
    fake_vm.total = 64 * 2**30

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict("sys.modules", {"funasr": fake_funasr}), \
         patch("psutil.virtual_memory", return_value=fake_vm):
        pipeline = AcousticProsodyPipeline("dummy.json", "dummy_out.json")

    assert pipeline.sensevoice_model is sensevoice_model
    assert fake_funasr.AutoModel.call_count == 2
    fake_funasr.AutoModel.assert_any_call(model="iic/SenseVoiceSmall", disable_update=True)

def test_sensevoice_stays_lazy_on_standard_memory_host():
    """Issue #1: on hosts below the 48 GB gate, SenseVoice is NOT constructed
    at init — it stays lazy so the Mac mini does not pay resident memory it
    may never use. The SER model identifier is resolved by the tier-per-host
    registry (`ml_dependencies.md` Resolved Issue #1) — on a 24 GB host that
    means the `small`-tier `iic/emotion2vec_plus_base` variant.
    """
    fake_funasr = MagicMock()
    emotion_model = MagicMock(name="emotion2vec")
    fake_funasr.AutoModel = MagicMock(return_value=emotion_model)

    fake_vm = MagicMock()
    fake_vm.total = 24 * 2**30

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict("sys.modules", {"funasr": fake_funasr}), \
         patch("psutil.virtual_memory", return_value=fake_vm):
        pipeline = AcousticProsodyPipeline("dummy.json", "dummy_out.json")

    assert pipeline.sensevoice_model is None
    fake_funasr.AutoModel.assert_called_once_with(
        model="iic/emotion2vec_plus_base", disable_update=True
    )

def test_eager_load_failure_falls_back_to_lazy():
    """Issue #1: eager-load is a pure optimization — if SenseVoice construction
    fails on a high-memory host, _init_models must not hard-fail (SenseVoice is
    supplementary). It falls back to the lazy path with sensevoice_model=None.
    """
    fake_funasr = MagicMock()
    emotion_model = MagicMock(name="emotion2vec")
    fake_funasr.AutoModel = MagicMock(
        side_effect=[emotion_model, OSError("SenseVoice weights corrupt")]
    )

    fake_vm = MagicMock()
    fake_vm.total = 64 * 2**30

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict("sys.modules", {"funasr": fake_funasr}), \
         patch("psutil.virtual_memory", return_value=fake_vm):
        pipeline = AcousticProsodyPipeline("dummy.json", "dummy_out.json")

    assert pipeline.sensevoice_model is None
    assert pipeline.funasr_available is True

# ------------------------------------------------------------------
#  Tunable config (Layer03cConfig)
# ------------------------------------------------------------------

def test_config_thresholds_override_classification():
    """Issue #5: overriding Layer03cConfig changes heuristic behavior without
    editing source. Here a stricter min_dominant_tone_score forces what would
    have been Alarming back down to Neutral.
    """
    cfg = Layer03cConfig(min_dominant_tone_score=2.0)
    pipeline = _make_pipeline()
    pipeline.config = cfg

    emotions = {"angry": 0.6, "fearful": 0.3, "neutral": 0.1}
    tone, scalar = pipeline._classify_acoustic_tone(emotions, max_amp_dbFS=-10.0, pitch_variance=0.0)
    assert tone == "Neutral"
    assert scalar == 0.0

def test_config_volume_cutoff_override_changes_alarming_bonus():
    """Lifting the high_volume_dbfs cutoff above the input dBFS removes the bonus."""
    pipeline = _make_pipeline()
    # With default cfg, -25 dBFS is below the -20 cutoff and gets no bonus.
    # Push the cutoff to -30 so -25 now exceeds it and earns the bonus.
    pipeline.config = Layer03cConfig(high_volume_dbfs=-30.0)
    emotions = {"angry": 0.4, "fearful": 0.2, "neutral": 0.4}
    tone, _ = pipeline._classify_acoustic_tone(emotions, max_amp_dbFS=-25.0, pitch_variance=0.0)
    assert tone == "Alarming"

# ------------------------------------------------------------------
#  Temp file lifecycle (Issue #2)
# ------------------------------------------------------------------

def test_wav_cleaned_up_on_processing_exception(tmp_path):
    """Issue #2: when librosa/funasr raise mid-pipeline, the temp wav file
    must still be removed via the try/finally cleanup. Otherwise the OS temp
    directory accumulates orphaned files on every failed video.
    """
    pipeline = _make_pipeline()
    fake_wav = "/tmp/prosody_unit_test.wav"
    cleanup_calls = []

    def fake_remove(path):
        cleanup_calls.append(path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated librosa OOM")

    pipeline._extract_audio_chunk = lambda v, s, e: fake_wav
    pipeline._extract_librosa_features = boom
    # _safe_remove is a staticmethod; patch on the class so the bound call
    # routes through our recorder.
    with patch.object(AcousticProsodyPipeline, "_safe_remove", staticmethod(fake_remove)):
        with pytest.raises(RuntimeError, match="simulated librosa OOM"):
            pipeline._process_task("dummy.mp4", {
                "task_id": "t_x",
                "task_temporal_metadata": {"task_reaction_window_sec": [0.0, 1.0]},
            })

    assert fake_wav in cleanup_calls, "Temp wav must be cleaned up even when processing raises"

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
