import json
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from dataset_acquisition.registry import scan_datasets
from filtering_and_labeling.pipeline import FilteringPipeline
from layer_03a_attention.pipeline import AttentionLayerPipeline
from layer_04_dehydrated_export.aggregator import DataAggregator
from layer_04_dehydrated_export.export import DehydratedExporter
from layer_04_dehydrated_export.huggingface_upload import upload_to_huggingface
import dataset_acquisition.synthetic.generator as gen_mod
from dataset_acquisition.synthetic.generator import SyntheticVideoGenerator


@pytest.fixture
def mock_datasets_dir(tmp_path):
    # Setup mock synthetic validation folder structure
    dataset_dir = tmp_path / "synthetic_validation"
    scenario_dir = dataset_dir / "interaction_3"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fake mp4
    video_file = scenario_dir / "abc123ef.mp4"
    with open(video_file, "wb") as f:
        f.write(b"fake mp4 video data")

    # Provenance sidecar (written by generator.py in production)
    sidecar = scenario_dir / "abc123ef.json"
    with open(sidecar, "w") as f:
        json.dump({"generator": "wan2.1-t2v", "generator_version": "14b-diffusers"}, f)

    return dataset_dir


def test_registry_scan_detects_synthetic(mock_datasets_dir, monkeypatch):
    # Mock config.DATASET_PATHS
    monkeypatch.setitem(config.DATASET_PATHS, "synthetic_validation", [mock_datasets_dir])
    
    registry = scan_datasets()
    
    # Find synthetic entry
    synthetic_entry = next((e for e in registry if e.get("id") == "synthetic_interaction_3_abc123ef"), None)
    assert synthetic_entry is not None
    assert synthetic_entry["synthetic"] is True
    assert synthetic_entry["scenario_tag"] == "interaction_3"
    assert synthetic_entry["prompt_hash"] == "abc123ef"
    assert synthetic_entry["generator"] == "wan2.1-t2v"
    assert synthetic_entry["generator_version"] == "14b-diffusers"
    assert synthetic_entry["expected_pass"] is True


def test_filtering_pipeline_bypasses_and_labels_synthetic(tmp_path, monkeypatch):
    input_manifest = tmp_path / "registry.json"
    output_manifest = tmp_path / "filtered_manifest.json"
    
    video_file = tmp_path / "dummy.mp4"
    with open(video_file, "w") as f:
        f.write("fake video content")
        
    # Write a fake registry entry for a synthetic video
    registry_data = [
        {
            "id": "synthetic_interaction_3_abc123ef",
            "dataset": "synthetic_validation",
            "file_path": str(video_file),
            "file_size": 1000,
            "created_at": 123456789,
            "synthetic": True,
            "scenario_tag": "interaction_3",
            "prompt_hash": "abc123ef",
            "generator": "wan2.1-t2v",
            "generator_version": "14b-diffusers",
            "expected_pass": True,
        }
    ]
    with open(input_manifest, "w") as f:
        json.dump(registry_data, f)
        
    # Mock cv2.VideoCapture to avoid opening actual file
    class MockCap:
        def isOpened(self): return True
        def get(self, prop):
            # 5 is FPS, 7 is FRAME_COUNT
            if prop == 5: return 30.0
            if prop == 7: return 150
            return 0.0
        def release(self): pass
        def read(self): return True, None
        
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: MockCap())
    
    # Mock SocialPresenceDetector and metadata loading
    monkeypatch.setattr("filtering_and_labeling.pipeline.SocialPresenceDetector", MagicMock())
    monkeypatch.setattr(FilteringPipeline, "_load_metadata", lambda self: {})
    
    # run_synthetic_qa=True: this test exercises the synthetic-handling code
    # path itself, which must still work when explicitly enabled. The env
    # default is off (June 12: Layer 1a parked — see docs/01a Unresolved #1).
    pipeline = FilteringPipeline(input_manifest, output_manifest, run_synthetic_qa=True)
    pipeline.initial_registry = registry_data  # set manually
    
    # Mock social_presence_filter to bypass YOLO run
    monkeypatch.setattr(pipeline, "social_presence_filter", lambda video_path: (
        [{"person_id": 0, "timestamps_sec": [0.0], "bounding_boxes": [[10, 10, 50, 50]]}], []
    ))
    
    # Mock process_video to bypass actual VLM logic
    # but still let it exercise process_video_vlm_pass and contextual_task_labeling
    orig_process_video_vlm_pass = pipeline.process_video_vlm_pass
    def mock_vlm_pass(entry):
        return orig_process_video_vlm_pass(entry)
    
    monkeypatch.setattr(pipeline, "process_video_vlm_pass", mock_vlm_pass)
    
    pipeline.run()
    
    # Verify the output manifest
    assert output_manifest.exists()
    with open(output_manifest, "r") as f:
        filtered = json.load(f)
        
    assert len(filtered) == 1
    out_entry = filtered[0]
    assert out_entry["video_id"] == "synthetic_interaction_3_abc123ef"
    assert out_entry["synthetic"] is True
    assert out_entry["scenario_tag"] == "interaction_3"
    assert out_entry["expected_pass"] is True
    assert len(out_entry["identified_tasks"]) == 1
    task = out_entry["identified_tasks"][0]
    assert task["task_label"] == "Synthetic validation task: interaction_3"
    assert task["task_id"] == "t_01"


def test_layer_03_skips_synthetic(tmp_path, monkeypatch):
    input_manifest = tmp_path / "filtered_manifest.json"
    output_result = tmp_path / "03a_attention_result.json"
    
    # Input manifest has both a synthetic and a non-synthetic video
    manifest_data = [
        {
            "video_id": "synthetic_interaction_3_abc123ef",
            "synthetic": True,
            "video_path": "dummy1.mp4",
        },
        {
            "video_id": "ego4d_real_video",
            "video_path": "dummy2.mp4",
        }
    ]
    with open(input_manifest, "w") as f:
        json.dump(manifest_data, f)
        
    # Mock select_device / torch device
    monkeypatch.setattr("layer_03a_attention.pipeline.select_device", lambda x: "cpu")
    
    pipeline = AttentionLayerPipeline(input_manifest, output_result)
    pipeline.gaze_pipeline = None
    # 03a Resolved Issue 6: fixture entries have no real video, so the
    # face-quality pre-pass would zero-score and (correctly) gate them out —
    # disable the clip gate for this synthetic-skip plumbing test.
    pipeline.enable_face_quality_gate = False

    # Mock process_video to return a dummy result for non-synthetic
    monkeypatch.setattr(pipeline, "process_video", lambda entry: {"video_id": entry["video_id"]})
    
    pipeline.run()
    
    # Assert that only the real video was processed and saved
    assert output_result.exists()
    with open(output_result, "r") as f:
        results = json.load(f)
        
    assert len(results) == 1
    assert results[0]["video_id"] == "ego4d_real_video"


def test_aggregator_excludes_synthetic(tmp_path, monkeypatch):
    manifest_path = tmp_path / "filtered_manifest.json"
    result_path = tmp_path / "03a_result.json"
    
    manifest_data = [
        {
            "video_id": "synthetic_interaction_3_abc123ef",
            "synthetic": True,
            "source_dataset": "synthetic_validation",
        },
        {
            "video_id": "ego4d_real_video",
            "source_dataset": "ego4d",
        }
    ]
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)
        
    # Real 03a result file won't have synthetic since 03a skipped it, but we can double check
    result_data = [
        {
            "video_id": "ego4d_real_video",
            "layer": "03a_attention",
            "tasks_analyzed": []
        }
    ]
    with open(result_path, "w") as f:
        json.dump(result_data, f)
        
    aggregator = DataAggregator(str(tmp_path))
    df = aggregator.aggregate()
    
    assert df is not None
    # Should only contain the real video, synthetic is gated out
    assert len(df) == 1
    assert df.iloc[0]["video_id"] == "ego4d_real_video"
    assert "synthetic_interaction_3_abc123ef" not in df["video_id"].values


def test_exporter_raises_on_synthetic_video_id(tmp_path):
    exporter = DehydratedExporter(str(tmp_path))
    
    # DataFrame with synthetic video ID
    df_synthetic = pd.DataFrame([
        {"video_id": "synthetic_interaction_3_abc123ef", "source_dataset": "synthetic_validation"}
    ])
    
    with pytest.raises(ValueError, match="Dehydration validation failed: DataFrame contains synthetic validation video IDs"):
        exporter.export_parquet(df_synthetic)
        
    # Normal DataFrame passes
    df_normal = pd.DataFrame([
        {"video_id": "ego4d_real_video", "source_dataset": "ego4d"}
    ])
    exporter.export_parquet(df_normal)
    assert (tmp_path / "social_metadata.parquet").exists()


def test_hf_upload_raises_on_synthetic_video_id(tmp_path):
    # Setup a Parquet file containing synthetic video ID
    parquet_path = tmp_path / "social_metadata.parquet"
    df_synthetic = pd.DataFrame([
        {"video_id": "synthetic_interaction_3_abc123ef", "source_dataset": "synthetic_validation"}
    ])
    df_synthetic.to_parquet(parquet_path)
    
    with pytest.raises(ValueError, match="HF upload validation failed: Parquet file contains synthetic validation video IDs"):
        upload_to_huggingface(str(tmp_path), repo_id="dummy/repo", token="dummy")


def _seed_export_bundle(tmp_path):
    """A minimal, non-synthetic export bundle ready to upload."""
    pd.DataFrame([{"video_id": "ego4d_real"}]).to_parquet(tmp_path / "social_metadata.parquet")
    (tmp_path / "export_metadata.json").write_text("{}")


def test_hf_upload_falls_back_to_credential_store(tmp_path):
    """token=None must resolve via get_token() (the `hf auth login` store, not
    just HF_TOKEN env) and proceed — previously it skipped unless HF_TOKEN was
    set in the environment."""
    _seed_export_bundle(tmp_path)
    mock_api = MagicMock()
    with patch("layer_04_dehydrated_export.huggingface_upload.HfApi", return_value=mock_api), \
         patch("layer_04_dehydrated_export.huggingface_upload.get_token", return_value="stored-token"):
        result = upload_to_huggingface(str(tmp_path), repo_id="dummy/repo")  # no token arg
    mock_api.create_repo.assert_called_once()
    assert mock_api.upload_file.call_count == 4
    assert result == ["social_metadata.parquet", "export_metadata.json", "README.md", "rehydrate_dataset.py"]


def test_hf_upload_skips_only_when_no_token_anywhere(tmp_path):
    """With no token arg AND get_token() empty, publishing is unconfigured ->
    skip (return None) without ever constructing HfApi."""
    _seed_export_bundle(tmp_path)
    with patch("layer_04_dehydrated_export.huggingface_upload.get_token", return_value=None), \
         patch("layer_04_dehydrated_export.huggingface_upload.HfApi") as mock_api_cls:
        result = upload_to_huggingface(str(tmp_path), repo_id="dummy/repo")
    assert result is None
    mock_api_cls.assert_not_called()


def test_hf_upload_raises_on_upload_failure(tmp_path):
    """A real upload failure must propagate, not be swallowed as a warning."""
    _seed_export_bundle(tmp_path)
    mock_api = MagicMock()
    mock_api.upload_file.side_effect = RuntimeError("network down")
    with patch("layer_04_dehydrated_export.huggingface_upload.HfApi", return_value=mock_api):
        with pytest.raises(RuntimeError, match="network down"):
            upload_to_huggingface(str(tmp_path), repo_id="dummy/repo", token="tok")


def test_hf_upload_raises_on_repo_auth_failure(tmp_path):
    """A present-but-rejected token (create_repo 401) must propagate, not skip."""
    _seed_export_bundle(tmp_path)
    mock_api = MagicMock()
    mock_api.create_repo.side_effect = RuntimeError("401 Unauthorized")
    with patch("layer_04_dehydrated_export.huggingface_upload.HfApi", return_value=mock_api):
        with pytest.raises(RuntimeError, match="401"):
            upload_to_huggingface(str(tmp_path), repo_id="dummy/repo", token="badtoken")


# ---------------------------------------------------------------------------
# Registry fallback when a synthetic clip has no provenance sidecar
# ---------------------------------------------------------------------------
def test_registry_scan_synthetic_without_sidecar_uses_fallback(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "synthetic_validation"
    scenario_dir = dataset_dir / "handoff"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "deadbeef.mp4").write_bytes(b"x")

    monkeypatch.setitem(config.DATASET_PATHS, "synthetic_validation", [dataset_dir])
    registry = scan_datasets()

    entry = next((e for e in registry if e.get("id") == "synthetic_handoff_deadbeef"), None)
    assert entry is not None
    assert entry["generator"] == "wan2.1-t2v"
    assert entry["generator_version"] == "unknown"


# ---------------------------------------------------------------------------
# Wan2.1 local generator backend (diffusers mocked — no weights, no render)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_wan(monkeypatch):
    """Mock the diffusers WanPipeline stack so tests never download weights or render."""
    mock_pipe = MagicMock(name="WanPipeline_instance")
    mock_result = MagicMock()
    mock_result.frames = [["frame0"]]
    mock_pipe.return_value = mock_result

    mock_wan_cls = MagicMock(name="WanPipeline")
    mock_wan_cls.from_pretrained.return_value = mock_pipe

    monkeypatch.setattr(gen_mod, "WanPipeline", mock_wan_cls)
    monkeypatch.setattr(gen_mod, "AutoencoderKLWan", MagicMock(name="AutoencoderKLWan"))
    monkeypatch.setattr(gen_mod, "UniPCMultistepScheduler", MagicMock(name="UniPCMultistepScheduler"))

    def _fake_export(frames, dest, fps=None, **kwargs):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"fake-rendered-mp4")
        return dest

    export_mock = MagicMock(name="export_to_video", side_effect=_fake_export)
    monkeypatch.setattr(gen_mod, "export_to_video", export_mock)
    return {"pipe": mock_pipe, "cls": mock_wan_cls, "export": export_mock}


def test_generator_cache_hit_skips_pipeline(tmp_path, mock_wan):
    gen = SyntheticVideoGenerator(output_dir=tmp_path)
    tag = next(iter(gen.scenarios))
    prompt_hash = gen.get_prompt_hash(gen.get_prompt(tag))

    cached = tmp_path / tag / f"{prompt_hash}.mp4"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"already-here")

    result = gen.generate_video(tag, seed=0)

    assert result == cached
    mock_wan["cls"].from_pretrained.assert_not_called()
    mock_wan["export"].assert_not_called()


def test_generator_cache_miss_renders_and_writes_sidecar(tmp_path, mock_wan, monkeypatch):
    gen = SyntheticVideoGenerator(output_dir=tmp_path)
    monkeypatch.setattr(gen, "resolve_model_id", lambda: "Wan-AI/Wan2.1-T2V-14B-Diffusers")
    tag = next(iter(gen.scenarios))
    prompt_hash = gen.get_prompt_hash(gen.get_prompt(tag))

    result = gen.generate_video(tag, seed=0)

    assert result == tmp_path / tag / f"{prompt_hash}.mp4"
    assert result.exists()

    assert mock_wan["export"].call_count == 1
    _, kwargs = mock_wan["export"].call_args
    assert kwargs.get("fps") == 16

    sidecar = tmp_path / tag / f"{prompt_hash}.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["generator"] == "wan2.1-t2v"
    assert meta["generator_version"] == "14b-diffusers"
    assert meta["seed"] == 0
    assert meta["fps"] == 16


def test_generator_unknown_scenario_tag_raises(tmp_path):
    gen = SyntheticVideoGenerator(output_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown scenario tag"):
        gen.generate_video("does_not_exist")


def test_generator_prefers_mps(tmp_path, monkeypatch):
    gen = SyntheticVideoGenerator(output_dir=tmp_path)
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    if gen_mod.torch is not None:
        monkeypatch.setattr(gen_mod.torch.backends.mps, "is_available", lambda: True)
        assert gen.select_device() == "mps"


def test_generator_default_output_dir_is_extreme_ssd():
    gen = SyntheticVideoGenerator()
    assert str(gen.output_dir).startswith("/Volumes/Extreme SSD/")
