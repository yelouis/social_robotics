"""Tests for the bystander-face-quality pre-filter (03b Resolved Issue #8).

Covers the gate predicate's threshold + fail-open semantics and the
`entry_filter` hook that lets the climax runner skip low-yield clips.
"""
import json

from src.shared.face_quality_prefilter import passes_face_quality, FIELD
from src.shared import climax_extraction as ce


# ------------------------------------------------------------------
#  Gate predicate: threshold logic
# ------------------------------------------------------------------

def test_passes_above_threshold():
    e = {FIELD: {"best_face_px": 140, "best_face_conf": 0.97, "n_face_frames": 5}}
    assert passes_face_quality(e) is True


def test_fails_below_px():
    e = {FIELD: {"best_face_px": 80, "best_face_conf": 0.97, "n_face_frames": 5}}
    assert passes_face_quality(e) is False


def test_fails_below_conf():
    e = {FIELD: {"best_face_px": 140, "best_face_conf": 0.6, "n_face_frames": 5}}
    assert passes_face_quality(e) is False


def test_fails_below_frames():
    e = {FIELD: {"best_face_px": 140, "best_face_conf": 0.97, "n_face_frames": 1}}
    assert passes_face_quality(e) is False


def test_zero_record_fails():
    """A present-but-zero record (checked, no resolvable face) must fail (skip)."""
    e = {FIELD: {"best_face_px": 0, "best_face_conf": 0.0, "n_face_frames": 0, "n_checked": 24}}
    assert passes_face_quality(e) is False


# ------------------------------------------------------------------
#  Gate predicate: fail-open + toggle semantics
# ------------------------------------------------------------------

def test_missing_field_fails_open():
    """An un-scored clip (no field) is processed, not silently dropped."""
    assert passes_face_quality({"video_id": "x"}) is True


def test_gate_disabled_passes_everything():
    e = {FIELD: {"best_face_px": 0, "best_face_conf": 0.0, "n_face_frames": 0}}
    assert passes_face_quality(e, enabled=False) is True


def test_custom_thresholds():
    e = {FIELD: {"best_face_px": 100, "best_face_conf": 0.75, "n_face_frames": 2}}
    assert passes_face_quality(e, min_px=80, min_conf=0.7, min_frames=2) is True
    assert passes_face_quality(e, min_px=120, min_conf=0.8, min_frames=3) is False


# ------------------------------------------------------------------
#  climax entry_filter hook
# ------------------------------------------------------------------

def test_climax_entry_filter_skips_rejected(tmp_path, monkeypatch):
    """`entry_filter` prevents climax from being computed for rejected entries:
    the rejected entry is never passed to the worker and its metadata is left
    untouched."""
    manifest = tmp_path / "m.json"
    entries = [
        {"video_id": "keep", "video_path": "keep.mp4",
         "identified_tasks": [{"task_id": "t", "task_start_sec": 0, "task_end_sec": 1,
                               "task_temporal_metadata": {}}]},
        {"video_id": "drop", "video_path": "drop.mp4",
         "identified_tasks": [{"task_id": "t", "task_start_sec": 0, "task_end_sec": 1,
                               "task_temporal_metadata": {}}]},
    ]
    manifest.write_text(json.dumps(entries))

    seen = []

    def fake_worker(args):
        # Layer 02b worker signature (June 30): (entry, face_verify).
        entry, _face_verify = args
        seen.append(entry["video_id"])
        entry["identified_tasks"][0]["task_temporal_metadata"] = {"task_climax_sec": 0.5}
        return entry, True

    import layer_02b_task_climax.pipeline as l02b
    monkeypatch.setattr(l02b, "_annotate_one_entry", fake_worker)

    n = ce.populate_climax_for_manifest(
        manifest, entry_filter=lambda e: e["video_id"] == "keep")

    assert seen == ["keep"], "only the kept entry should reach the worker"
    assert n == 1
    data = json.loads(manifest.read_text())
    assert data[0]["identified_tasks"][0]["task_temporal_metadata"] == {"task_climax_sec": 0.5}
    # The rejected entry's metadata is left empty (climax never paid).
    assert data[1]["identified_tasks"][0]["task_temporal_metadata"] == {}


def test_climax_no_filter_processes_all(tmp_path, monkeypatch):
    """Default (entry_filter=None) is unchanged: every entry needing climax runs."""
    manifest = tmp_path / "m.json"
    entries = [
        {"video_id": v, "video_path": f"{v}.mp4",
         "identified_tasks": [{"task_id": "t", "task_start_sec": 0, "task_end_sec": 1,
                               "task_temporal_metadata": {}}]}
        for v in ("a", "b")
    ]
    manifest.write_text(json.dumps(entries))

    seen = []

    def fake_worker(args):
        # Layer 02b worker signature (June 30): (entry, face_verify).
        entry, _face_verify = args
        seen.append(entry["video_id"])
        return entry, True

    import layer_02b_task_climax.pipeline as l02b
    monkeypatch.setattr(l02b, "_annotate_one_entry", fake_worker)
    n = ce.populate_climax_for_manifest(manifest)
    assert sorted(seen) == ["a", "b"]
    assert n == 2
