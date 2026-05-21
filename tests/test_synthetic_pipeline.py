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
    assert synthetic_entry["generator"] == "veo-2"
    assert synthetic_entry["generator_version"] == "2.0-stable"
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
            "generator": "veo-2",
            "generator_version": "2.0-stable",
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
    
    pipeline = FilteringPipeline(input_manifest, output_manifest)
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
