import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.layer_03b_reasonable_emotion.pipeline import (
    ReasonableEmotionPipeline,
    ExpectationSchema,
    TransitionEvalSchema,
    DEFAULT_EXPECTATIONS,
)
import numpy as np

@pytest.fixture
def dummy_manifest(tmp_path):
    manifest_path = tmp_path / "filtered_manifest.json"
    data = [
        {
            "video_id": "test_video_1",
            "video_path": "dummy.mp4",  # Will be mocked
            "identified_tasks": [
                {
                    "task_id": "t_01",
                    "task_label": "Juggling apples",
                    "task_temporal_metadata": {
                        "task_climax_sec": 6.2,
                        "task_reaction_window_sec": [6.2, 8.2]
                    }
                }
            ],
            "bystander_detections": [
                {
                    "person_id": 0,
                    "timestamps_sec": [6.2, 6.7, 7.2, 7.7, 8.2],
                    "bounding_boxes": [
                        [100, 100, 200, 200]
                    ] * 5
                }
            ]
        }
    ]
    with open(manifest_path, 'w') as f:
        json.dump(data, f)
    return manifest_path

@pytest.fixture
def dummy_attention(tmp_path):
    att_path = tmp_path / "attention.json"
    data = [
        {
            "video_id": "test_video_1",
            "per_person": [
                {
                    "person_id": 0,
                    "average_attention_score": 0.8
                }
            ]
        }
    ]
    with open(att_path, 'w') as f:
        json.dump(data, f)
    return att_path

# ------------------------------------------------------------------
#  Pydantic schema validation tests
# ------------------------------------------------------------------

def test_expectation_schema_validates_valid_input():
    """ExpectationSchema accepts well-formed LLM output."""
    data = {
        "positive_emotions": ["joy", "excitement"],
        "negative_emotions": ["anger"],
        "neutral_baseline": ["neutral"]
    }
    schema = ExpectationSchema(**data)
    assert schema.positive_emotions == ["joy", "excitement"]

def test_expectation_schema_coerces_single_string():
    """ExpectationSchema coerces a bare string into a list."""
    data = {
        "positive_emotions": "joy",
        "negative_emotions": ["anger"],
        "neutral_baseline": ["neutral"]
    }
    schema = ExpectationSchema(**data)
    assert schema.positive_emotions == ["joy"]

def test_expectation_schema_rejects_missing_key():
    """ExpectationSchema raises ValidationError on missing required fields."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ExpectationSchema(positive_emotions=["joy"])

def test_transition_eval_schema_validates():
    """TransitionEvalSchema accepts valid classified_direction literals."""
    data = {"classified_direction": "positive", "reasoning": "test"}
    schema = TransitionEvalSchema(**data)
    assert schema.classified_direction == "positive"

def test_transition_eval_schema_rejects_bad_direction():
    """TransitionEvalSchema rejects invalid direction values."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TransitionEvalSchema(classified_direction="maybe", reasoning="test")

# ------------------------------------------------------------------
#  LLM retry + fallback tests
# ------------------------------------------------------------------

def test_expectation_fallback_when_ollama_unavailable():
    """When Ollama is not available, expectations fall back to defaults."""
    pipeline = ReasonableEmotionPipeline("dummy.json", "dummy_out.json")
    pipeline.ollama_available = False
    result = pipeline._llm_generate_expectations("Juggling apples")
    assert result == DEFAULT_EXPECTATIONS

def test_transition_fallback_when_ollama_unavailable():
    """When Ollama is not available, transition eval falls back to rule-based."""
    pipeline = ReasonableEmotionPipeline("dummy.json", "dummy_out.json")
    pipeline.ollama_available = False
    result = pipeline._llm_evaluate_transition(
        "Test", DEFAULT_EXPECTATIONS, "", "neutral", "joy"
    )
    assert result["classified_direction"] == "positive"

def test_rule_based_classify():
    """Rule-based fallback correctly classifies by set membership."""
    result = ReasonableEmotionPipeline._rule_based_classify(
        "anger", DEFAULT_EXPECTATIONS
    )
    assert result["classified_direction"] == "negative"

    result = ReasonableEmotionPipeline._rule_based_classify(
        "surprise", DEFAULT_EXPECTATIONS
    )
    assert result["classified_direction"] == "neutral"

# ------------------------------------------------------------------
#  Math verification tests
# ------------------------------------------------------------------

def test_late_stage_weighted_math():
    """Verify the Late-Stage Weighted Average formula against hand-calculated values.

    Setup (climax = 6.2):
      Mock timeseries: neutral(6.2) -> anger(6.7) -> anger(7.7) -> joy(8.2)

      Slice 1: neutral -> anger  | window [6.2, 6.7] | negative  | scalar=-1.0
        D_1 = 0.5, W_1 = max(0.1, 6.2-6.2) = 0.1
      Slice 2: anger -> anger    | window [6.7, 7.7] | negative  | scalar=0 (magnitude=0)
        D_2 = 1.0, W_2 = max(0.1, 6.7-6.2) = 0.5
      Slice 3: anger -> joy      | window [7.7, 8.2] | positive  | scalar=+1.0
        D_3 = 0.5, W_3 = max(0.1, 7.7-6.2) = 1.5

      Numerator = (-1.0*0.5*0.1) + (0.0*1.0*0.5) + (1.0*0.5*1.5)
               = -0.05 + 0.0 + 0.75 = 0.70
      Denominator = (0.5*0.1) + (1.0*0.5) + (0.5*1.5)
                  = 0.05 + 0.50 + 0.75 = 1.30
      Score = 0.70 / 1.30 ≈ 0.5385 -> rounded to 0.54
    """
    pipeline = ReasonableEmotionPipeline("dummy.json", "dummy_out.json")
    pipeline.ollama_available = False  # Use rule-based fallback for determinism

    bystander = {
        "person_id": 0,
        "timestamps_sec": [6.2, 8.2],
        "bounding_boxes": [[0, 0, 10, 10], [0, 0, 10, 10]]
    }

    task = {
        "task_id": "t_math",
        "task_label": "Test task",
        "task_temporal_metadata": {
            "task_climax_sec": 6.2,
            "task_reaction_window_sec": [6.2, 8.2]
        }
    }

    def mock_sample_emotions(cap, fps, start_sec, end_sec, bystander):
        return [
            {"t": 6.2, "emotion": "neutral", "magnitude": 1.0},
            {"t": 6.7, "emotion": "anger", "magnitude": 1.0},
            {"t": 7.7, "emotion": "anger", "magnitude": 0.0},
            {"t": 8.2, "emotion": "joy", "magnitude": 1.0}
        ]
    pipeline._sample_emotions = mock_sample_emotions

    result = pipeline._process_task(None, 30, task, [bystander], "test_video_1")

    assert result is not None

    slices = result['per_person'][0]['temporal_slices']
    assert len(slices) == 3

    # Verify slice classifications
    assert slices[0]['classified_direction'] == "negative"
    assert slices[0]['slice_success_scalar'] == -1.0
    assert slices[1]['classified_direction'] == "negative"
    assert slices[1]['slice_success_scalar'] == 0.0  # magnitude is 0
    assert slices[2]['classified_direction'] == "positive"
    assert slices[2]['slice_success_scalar'] == 1.0

    # Verify the final weighted average math
    person_score = result['per_person'][0]['late_stage_weighted_success_score']
    expected = round(0.70 / 1.30, 2)  # 0.54
    assert person_score == expected, f"Expected {expected}, got {person_score}"

    assert result['task_aggregate_score'] == expected

def test_surprise_classified_as_neutral():
    """Verify that 'surprise' is classified as neutral, matching the doc schema example."""
    pipeline = ReasonableEmotionPipeline("dummy.json", "dummy_out.json")
    pipeline.ollama_available = False

    bystander = {
        "person_id": 0,
        "timestamps_sec": [1.0, 2.0],
        "bounding_boxes": [[0, 0, 10, 10], [0, 0, 10, 10]]
    }

    task = {
        "task_id": "t_surprise",
        "task_label": "Test surprise",
        "task_temporal_metadata": {
            "task_climax_sec": 1.0,
            "task_reaction_window_sec": [1.0, 2.0]
        }
    }

    def mock_sample_emotions(cap, fps, start_sec, end_sec, bystander):
        return [
            {"t": 1.0, "emotion": "neutral", "magnitude": 0.85},
            {"t": 1.5, "emotion": "surprise", "magnitude": 0.85}
        ]
    pipeline._sample_emotions = mock_sample_emotions

    result = pipeline._process_task(None, 30, task, [bystander], "test_video_2")

    assert result is not None
    slices = result['per_person'][0]['temporal_slices']
    assert slices[0]['transition_pair'] == ["neutral", "surprise"]
    assert slices[0]['classified_direction'] == "neutral"
    assert slices[0]['slice_success_scalar'] == 0.0

# ------------------------------------------------------------------
#  End-to-end schema conformance
# ------------------------------------------------------------------

def test_schema_conformance(dummy_manifest, dummy_attention, tmp_path, monkeypatch):
    out_path = tmp_path / "out.json"
    pipeline = ReasonableEmotionPipeline(dummy_manifest, out_path, dummy_attention)
    pipeline.ollama_available = False

    # Mock cv2 to pretend video opens and has frames
    class MockCap:
        def isOpened(self): return True
        def get(self, prop): return 30.0 if prop == 5 else 300  # fps, total_frames
        def release(self): pass
        def read(self): return True, np.zeros((480, 640, 3), dtype=np.uint8)
        def set(self, prop, val): pass

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda x: MockCap())

    # Monkeypatch video exists check
    monkeypatch.setattr(Path, "exists", lambda x: True)

    pipeline.run()

    # Now verify the output file
    with open(out_path, 'r') as f:
        results = json.load(f)

    assert len(results) == 1
    res = results[0]
    assert res['video_id'] == "test_video_1"
    assert res['layer'] == "03b_reasonable_emotion"
    assert "tasks_analyzed" in res
    assert len(res["tasks_analyzed"]) == 1

    task_res = res["tasks_analyzed"][0]
    assert "task_id" in task_res
    assert "task_label" in task_res
    assert "per_person" in task_res
    assert "task_aggregate_score" in task_res

    person = task_res["per_person"][0]
    assert person["person_id"] == 0
    assert "late_stage_weighted_success_score" in person
    assert "temporal_slices" in person
    assert len(person["temporal_slices"]) > 0

    slice1 = person["temporal_slices"][0]
    assert "slice_id" in slice1
    assert "window_sec" in slice1
    assert "transition_pair" in slice1
    assert "classified_direction" in slice1
    assert "slice_success_scalar" in slice1
