"""Unified per-segment export tests (docs/06 Issue 2): the cross-layer join
rules (segment attribution, null-reasons, genuine-track filter, ambient
prosody scope) and the confidence-gated QA renderer."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layer_04_dehydrated_export.segment_dataset import (  # noqa: E402
    build_segment_rows,
    build_segment_dataset,
    render_qa_pairs,
)


def _manifest_entry():
    return {
        "id": "vid1",
        "identified_tasks": [{
            "task_id": "t0",
            "task_label": "Playing cards",
            "task_temporal_metadata": {
                "reaction_segments": [
                    {"task_climax_sec": 10.0, "task_reaction_window_sec": [9.0, 13.0],
                     "segment_action_caption": "hands a card to the player opposite",
                     "segment_face_px": 200, "cluster_detection_count": 9},
                    {"task_climax_sec": 50.0, "task_reaction_window_sec": [49.0, 53.0],
                     "segment_face_px": 0, "cluster_detection_count": 4},
                ],
                "n_reaction_segments": 2,
            },
        }],
    }


def _results():
    return {
        "03a": {"vid1": {"video_id": "vid1", "per_person": [
            {"person_id": 1, "attention_trace": [
                {"t": 9.5, "score": 0.9, "head_pitch_rad": 0.1},
                {"t": 10.5, "score": 0.7, "head_pitch_rad": None},
                {"t": 60.0, "score": 0.2},  # outside both windows
            ]},
        ]}},
        "03e": {"vid1": {"video_id": "vid1", "tasks_analyzed": [
            {"task_id": "t0", "segment_index": 0, "per_person": [
                {"person_id": 1, "gesture_detected": "affirming_nod", "confidence": 0.9,
                 "interpolated_fraction": 0.1},
                {"person_id": -5, "gesture_detected": "affirming_nod", "confidence": 0.9},
            ]},
        ]}},
        "03c": {"vid1": {"video_id": "vid1", "tasks_analyzed": [
            {"task_id": "t0", "segment_index": 0, "prosody_scalar": 0.6,
             "classified_acoustic_tone": "Soothing",
             "prosody_metrics": {"audio_present": True}},
        ]}},
    }


def _rows():
    return build_segment_rows([_manifest_entry()], _results())


def test_join_keys_and_base_fields():
    rows = _rows()
    seg0 = [r for r in rows if r["segment_index"] == 0]
    assert len(seg0) == 1 and seg0[0]["person_id"] == 1   # negative id dropped
    assert seg0[0]["segment_action_caption"].startswith("hands a card")
    assert seg0[0]["is_control"] is False


def test_attention_sliced_to_window():
    r = next(r for r in _rows() if r["segment_index"] == 0)
    assert r["attention_n_trace"] == 2            # 9.5 & 10.5 in [9,13]; 60.0 out
    assert r["attention_n_head_pose"] == 1
    assert abs(r["attention_mean_score"] - 0.8) < 1e-6
    assert r["attention_null_reason"] is None


def test_null_reasons_explicit():
    rows = _rows()
    seg1 = next(r for r in rows if r["segment_index"] == 1)
    # segment 1: no layer measured it -> person-less row with reasons, not silence
    assert seg1["person_id"] is None
    assert seg1["gesture_null_reason"] == "unmeasured_by_layer"
    assert seg1["proxemic_null_reason"] == "layer_not_run"   # no 03d result at all
    assert seg1["attention_null_reason"] == "person_absent"  # person_id None


def test_prosody_is_ambient_segment_scoped():
    r = next(r for r in _rows() if r["segment_index"] == 0)
    assert r["prosody_scalar"] == 0.6 and r["prosody_scope"] == "ambient"
    assert r["audio_present"] is True


def test_segment_unattributed_fallback():
    results = _results()
    # strip segment_index (pre-July-2 result): 2 segments -> ambiguous
    del results["03e"]["vid1"]["tasks_analyzed"][0]["segment_index"]
    rows = build_segment_rows([_manifest_entry()], results)
    seg0 = next(r for r in rows if r["segment_index"] == 0)
    assert seg0["gesture_null_reason"] == "segment_unattributed"

    # single-segment task -> unambiguous, joined at task level
    entry = _manifest_entry()
    entry["identified_tasks"][0]["task_temporal_metadata"]["reaction_segments"] = \
        entry["identified_tasks"][0]["task_temporal_metadata"]["reaction_segments"][:1]
    rows = build_segment_rows([entry], results)
    seg0 = next(r for r in rows if r["segment_index"] == 0)
    assert seg0["gesture_detected"] == "affirming_nod"


def test_qa_pairs_gated_by_confidence():
    r = next(r for r in _rows() if r["segment_index"] == 0)
    pairs = render_qa_pairs(r)
    channels = {p["channel"] for p in pairs}
    assert {"attention", "gesture", "prosody"} <= channels
    gesture = next(p for p in pairs if p["channel"] == "gesture")
    assert gesture["answer"] == "nods in affirmation"
    assert "hands a card" in gesture["question"]          # caption contextualizes
    prosody = next(p for p in pairs if p["channel"] == "prosody")
    assert "ambient" in prosody["question"].lower()

    # low-confidence gesture renders nothing
    r2 = dict(r, gesture_confidence=0.2)
    assert all(p["channel"] != "gesture" for p in render_qa_pairs(r2))
    # 'unclear' caption falls back to the generic context phrase
    r3 = dict(r, segment_action_caption="unclear")
    assert render_qa_pairs(r3)[0]["question"].startswith("During this moment")


def test_qa_caption_composition():
    r = next(r for r in _rows() if r["segment_index"] == 0)
    # bare gerund -> copula inserted (July 3 E2E finding)
    q = render_qa_pairs(dict(r, segment_action_caption="listening"))[0]["question"]
    assert q.startswith("While the camera wearer is listening,")
    # third-person + legacy trailing period/capital -> composed cleanly
    q = render_qa_pairs(dict(r, segment_action_caption="Hands cards to person on left."))[0]["question"]
    assert q.startswith("While the camera wearer hands cards to person on left,")


def test_end_to_end_files(tmp_path):
    (tmp_path / "filtered_manifest.json").write_text(json.dumps([_manifest_entry()]))
    for lid, per_video in _results().items():
        name = {"03a": "03a_attention_result.json",
                "03e": "03e_affirmation_gesture_result.json",
                "03c": "03c_acoustic_prosody_result.json"}[lid]
        (tmp_path / name).write_text(json.dumps(list(per_video.values())))
    s = build_segment_dataset(tmp_path, git_sha="testsha")
    assert s["n_rows"] == 2 and s["n_segments"] == 2 and s["n_clips"] == 1
    assert s["n_qa_pairs"] >= 3
    out = tmp_path / "segment_dataset"
    assert (out / "segment_rows.parquet").exists()
    lines = (out / "qa_pairs.jsonl").read_text().strip().splitlines()
    assert len(lines) == s["n_qa_pairs"]
    meta = json.loads((out / "segment_export_metadata.json").read_text())
    assert meta["schema_version"] == 1 and meta["pipeline_git_sha"] == "testsha"
    assert meta["layers_joined"] == ["03a", "03c", "03e"]
