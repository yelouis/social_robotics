"""Unified per-SEGMENT export + QA-pair emitter (docs/06 Issue 2).

The per-layer exports (`per_layer.py`) publish one dataset per layer, one row
per VIDEO. VLA fine-tuning and the social benchmark need the opposite shape:
**one row per `(video_id, task_id, segment_index, person_id)`, joined across
03a–03f**, so each row is a complete (context, action, reaction) training item
with per-channel confidences and explicit null-reasons.

The join encodes the cross-layer rules ONCE (docs/03 guardrails):
  * genuine-track filter — negative `person_id`s (untracked Node-02 fragments)
    are dropped;
  * segment attribution — layer rows carry `segment_index` (emitted by every
    03x since July 2); a row from an OLDER result file lacks it and is joined
    at task level only when the task has a single segment, else the channel is
    marked `segment_unattributed` rather than guessed;
  * per-channel null-reasons — `layer_not_run` (no result file / clip absent),
    `unmeasured_by_layer` (clip present, this segment/person not scored — e.g.
    03e's distinct-window dedup collapsed it), never silent NaN;
  * 03c prosody is AMBIENT audio (no speaker separation): its values repeat
    across the segment's person rows with `prosody_scope = "ambient"` and its
    QA pair says "around", never "by the bystander".

Outputs (all local, dehydrated — no pixels):
  * `segment_rows.parquet` — the joined table;
  * `qa_pairs.jsonl` — VQA co-training pairs rendered from confident channels
    (one JSON object per line: question, answer, channel, keys, confidence);
  * `segment_export_metadata.json` — schema version + provenance + counts.

CLI:
    python -m layer_04_dehydrated_export.segment_dataset <results_dir> \
        [--manifest M] [--out-dir D]
(results_dir holds `filtered_manifest.json` + the canonical `03*_result.json`
files, exactly like the per-layer publisher.)
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SCHEMA_VERSION = 1

# QA-pair confidence gates: a pair is only rendered when the channel's own
# confidence field clears these (a benchmark item must not inherit a guess).
QA_MIN_GESTURE_CONF = 0.6
QA_MIN_PROXEMIC_CONF = 0.3
QA_MIN_PROSODY_ABS = 0.25
QA_ATTENTION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_results(results_dir: Path) -> Dict[str, Dict[str, dict]]:
    """{layer_id ('03a'…): {video_id: record}} from canonical 03*_result.json."""
    out: Dict[str, Dict[str, dict]] = {}
    for p in sorted(results_dir.glob("03*_result.json")):
        layer_id = p.stem.split("_", 1)[0]
        try:
            with open(p) as f:
                records = json.load(f)
        except Exception:
            continue
        out[layer_id] = {r.get("video_id"): r for r in records if isinstance(r, dict)}
    return out


def _rows_for_segment(record: Optional[dict], task_id, segment_index,
                      n_segments: int) -> Tuple[Optional[list], Optional[str]]:
    """Layer `tasks_analyzed` rows for one segment -> (rows, null_reason).

    Exact (task_id, segment_index) match wins. A row WITHOUT segment_index
    (pre-July-2 result file) is used only when the task has a single segment
    (unambiguous); otherwise the channel is `segment_unattributed`."""
    if record is None:
        return None, "layer_not_run"
    if record.get("skipped_reason"):
        return None, f"unmeasured_by_layer:{record['skipped_reason']}"
    exact, unattributed = [], []
    for ta in record.get("tasks_analyzed") or []:
        if ta.get("task_id") != task_id:
            continue
        si = ta.get("segment_index")
        if si == segment_index:
            exact.append(ta)
        elif si is None:
            unattributed.append(ta)
    if exact:
        return exact, None
    if unattributed and n_segments == 1 and segment_index == 0:
        return unattributed, None
    if unattributed:
        return None, "segment_unattributed"
    return None, "unmeasured_by_layer"


def _person_entry(rows: Optional[list], person_id) -> Optional[dict]:
    for ta in rows or []:
        for p in ta.get("per_person") or []:
            if p.get("person_id") == person_id:
                return p
    return None


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------
def build_segment_rows(manifest: list, results: Dict[str, Dict[str, dict]]) -> List[dict]:
    rows: List[dict] = []
    for entry in manifest:
        video_id = entry.get("id", entry.get("video_id"))
        att_record = (results.get("03a") or {}).get(video_id)

        for task in entry.get("identified_tasks") or []:
            task_id = task.get("task_id", "unknown")
            meta = task.get("task_temporal_metadata") or {}
            segments = meta.get("reaction_segments") or []
            n_segments = len(segments)

            for seg_idx, seg in enumerate(segments):
                w = seg.get("task_reaction_window_sec") or [None, None]
                ego = seg.get("wearer_egomotion_proxy") or {}
                base = {
                    "video_id": video_id,
                    "task_id": task_id,
                    "segment_index": seg_idx,
                    "task_label": task.get("task_label"),
                    "segment_action_caption": seg.get("segment_action_caption"),
                    "is_control": bool(seg.get("is_control", False)),
                    "control_type": seg.get("control_type"),
                    "climax_sec": seg.get("task_climax_sec"),
                    "window_start_sec": w[0],
                    "window_end_sec": w[1],
                    "segment_face_px": seg.get("segment_face_px"),
                    "cluster_detection_count": seg.get("cluster_detection_count"),
                    # Wearer pseudo-action features (docs/06 Issue 5).
                    "wearer_n_hand_detections": (len(seg["wearer_hand_detections"])
                                                 if seg.get("wearer_hand_detections") is not None
                                                 else None),
                    "wearer_egomotion_mean": ego.get("mean_frame_diff"),
                    "wearer_egomotion_peak": ego.get("peak_frame_diff"),
                }

                # Per-layer segment rows (+ null reason when absent).
                seg_rows = {lid: _rows_for_segment((results.get(lid) or {}).get(video_id),
                                                   task_id, seg_idx, n_segments)
                            for lid in ("03b", "03c", "03d", "03e", "03f")}

                # 03c prosody is segment-scoped ambient audio (identical on
                # every person row of the segment).
                c_rows, c_reason = seg_rows["03c"]
                prosody = {"prosody_scalar": None, "prosody_tone": None,
                           "audio_present": None, "prosody_scope": "ambient",
                           "prosody_null_reason": c_reason}
                if c_rows:
                    c = c_rows[0]
                    pm = c.get("prosody_metrics") or {}
                    prosody.update(prosody_scalar=c.get("prosody_scalar"),
                                   prosody_tone=c.get("classified_acoustic_tone"),
                                   audio_present=pm.get("audio_present"),
                                   prosody_null_reason=None)

                # 03f carries a task/segment-level wearer-context score.
                f_rows, _ = seg_rows["03f"]
                ego_chaos = f_rows[0].get("ego_kinetic_chaos_score") if f_rows else None

                # Person universe: every positive person_id any layer measured
                # for this segment (genuine-track filter: negatives dropped).
                persons = set()
                for lid in ("03b", "03d", "03e", "03f"):
                    lrows, _ = seg_rows[lid]
                    for ta in lrows or []:
                        for p in ta.get("per_person") or []:
                            pid = p.get("person_id")
                            if pid is not None and pid >= 0:
                                persons.add(pid)

                if not persons:
                    # Keep the segment visible (context + null reasons), person-less.
                    rows.append({**base, "person_id": None, **prosody,
                                 "ego_kinetic_chaos_score": ego_chaos,
                                 **_attention_channel(att_record, None, w),
                                 **_emotion_channel(seg_rows["03b"], None),
                                 **_proxemic_channel(seg_rows["03d"], None),
                                 **_gesture_channel(seg_rows["03e"], None),
                                 **_resonance_channel(seg_rows["03f"], None)})
                    continue

                for pid in sorted(persons):
                    rows.append({**base, "person_id": pid, **prosody,
                                 "ego_kinetic_chaos_score": ego_chaos,
                                 **_attention_channel(att_record, pid, w),
                                 **_emotion_channel(seg_rows["03b"], pid),
                                 **_proxemic_channel(seg_rows["03d"], pid),
                                 **_gesture_channel(seg_rows["03e"], pid),
                                 **_resonance_channel(seg_rows["03f"], pid)})
    return rows


# --- per-channel extractors (each returns its columns + *_null_reason) ---
def _attention_channel(att_record, person_id, window) -> dict:
    out = {"attention_mean_score": None, "attention_n_trace": None,
           "attention_n_head_pose": None, "attention_null_reason": None}
    if att_record is None:
        out["attention_null_reason"] = "layer_not_run"
        return out
    person = next((p for p in att_record.get("per_person") or []
                   if p.get("person_id") == person_id), None)
    if person is None:
        out["attention_null_reason"] = "person_absent"
        return out
    w0, w1 = (window or [None, None])[:2]
    if w0 is None:
        out["attention_null_reason"] = "no_window"
        return out
    # 03a is clip-level (climax-independent): slice its trace to this window.
    in_win = [x for x in person.get("attention_trace") or [] if w0 <= x.get("t", -1) <= w1]
    if not in_win:
        out["attention_null_reason"] = "no_trace_in_window"
        return out
    scores = [x.get("score") for x in in_win if x.get("score") is not None]
    out.update(
        attention_mean_score=round(sum(scores) / len(scores), 3) if scores else None,
        attention_n_trace=len(in_win),
        attention_n_head_pose=sum(1 for x in in_win if x.get("head_pitch_rad") is not None),
    )
    return out


def _emotion_channel(rows_reason, person_id) -> dict:
    rows, reason = rows_reason
    out = {"emotion_task_score": None, "emotion_person_score": None,
           "emotion_null_reason": reason}
    if not rows:
        return out
    out["emotion_task_score"] = rows[0].get("task_aggregate_score")
    p = _person_entry(rows, person_id)
    if p is None:
        out["emotion_null_reason"] = "person_absent"
    else:
        out["emotion_person_score"] = p.get("late_stage_weighted_success_score")
    return out


def _proxemic_channel(rows_reason, person_id) -> dict:
    rows, reason = rows_reason
    out = {"proxemic_vector": None, "proxemic_action": None,
           "proxemic_confidence": None, "proxemic_null_reason": reason}
    p = _person_entry(rows, person_id)
    if rows and p is None:
        out["proxemic_null_reason"] = "person_absent"
    elif p:
        out.update(proxemic_vector=p.get("proxemic_vector"),
                   proxemic_action=p.get("classified_action"),
                   proxemic_confidence=p.get("proxemic_confidence"))
    return out


def _gesture_channel(rows_reason, person_id) -> dict:
    rows, reason = rows_reason
    out = {"gesture_detected": None, "gesture_confidence": None,
           "gesture_interpolated_fraction": None, "gesture_null_reason": reason}
    p = _person_entry(rows, person_id)
    if rows and p is None:
        out["gesture_null_reason"] = "person_absent"
    elif p:
        out.update(gesture_detected=p.get("gesture_detected"),
                   gesture_confidence=p.get("confidence"),
                   gesture_interpolated_fraction=p.get("interpolated_fraction"))
    return out


def _resonance_channel(rows_reason, person_id) -> dict:
    rows, reason = rows_reason
    out = {"motor_resonance_detected": None, "empathy_scalar": None,
           "mirroring_detected": None, "resonance_null_reason": reason}
    p = _person_entry(rows, person_id)
    if rows and p is None:
        out["resonance_null_reason"] = "person_absent"
    elif p:
        out.update(motor_resonance_detected=p.get("motor_resonance_detected"),
                   empathy_scalar=p.get("empathy_scalar"),
                   mirroring_detected=p.get("mirroring_detected"))
    return out


# ---------------------------------------------------------------------------
# QA-pair rendering (VQA co-training / benchmark Tier 1)
# ---------------------------------------------------------------------------
def _context_phrase(row) -> str:
    cap = row.get("segment_action_caption")
    if cap and cap not in ("unclear",):
        if cap == "conversation":
            return "While the camera wearer is in a conversation"
        return f"While the camera wearer {cap}"
    return "During this moment"


def render_qa_pairs(row: dict) -> List[dict]:
    """Render QA pairs from the CONFIDENT channels of one joined row. A channel
    below its confidence gate (or null) renders nothing — a training/benchmark
    pair must never inherit a low-confidence guess."""
    keys = {k: row.get(k) for k in
            ("video_id", "task_id", "segment_index", "person_id")}
    ctx = _context_phrase(row)
    pairs: List[dict] = []

    score = row.get("attention_mean_score")
    if score is not None:
        yes = score >= QA_ATTENTION_THRESHOLD
        pairs.append({**keys, "channel": "attention",
                      "question": f"{ctx}, is the bystander paying attention to them?",
                      "answer": "yes" if yes else "no",
                      "confidence": round(abs(score - QA_ATTENTION_THRESHOLD) * 2, 3)})

    g, gc = row.get("gesture_detected"), row.get("gesture_confidence") or 0.0
    if g in ("affirming_nod", "negating_shake") and gc >= QA_MIN_GESTURE_CONF:
        pairs.append({**keys, "channel": "gesture",
                      "question": f"{ctx}, how does the bystander respond with their head?",
                      "answer": "nods in affirmation" if g == "affirming_nod"
                                else "shakes their head in negation",
                      "confidence": gc})

    a, pc = row.get("proxemic_action"), row.get("proxemic_confidence") or 0.0
    if a in ("Approach_Intervention", "Avoidance") and pc >= QA_MIN_PROXEMIC_CONF:
        pairs.append({**keys, "channel": "proxemics",
                      "question": f"{ctx}, does the bystander move toward or away from them?",
                      "answer": "moves toward them" if a == "Approach_Intervention"
                                else "moves away from them",
                      "confidence": pc})

    ps = row.get("prosody_scalar")
    if row.get("audio_present") and ps is not None and abs(ps) >= QA_MIN_PROSODY_ABS:
        pairs.append({**keys, "channel": "prosody",
                      "question": f"{ctx}, what is the tone of the voices AROUND them "
                                  "(ambient audio, not attributed to one speaker)?",
                      "answer": "positive or soothing" if ps > 0 else "negative or alarming",
                      "confidence": round(abs(ps), 3)})

    if row.get("motor_resonance_detected") and (row.get("empathy_scalar") or 0) > 0:
        pairs.append({**keys, "channel": "motor_resonance",
                      "question": f"{ctx}, does the bystander flinch in sympathy with the "
                                  "wearer's sudden movement?",
                      "answer": "yes",
                      "confidence": row.get("empathy_scalar")})
    return pairs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_segment_dataset(results_dir, manifest_path=None, out_dir=None,
                          git_sha: str = "unknown") -> dict:
    """Join manifest segments with every 03*_result.json in ``results_dir``,
    write `segment_rows.parquet` + `qa_pairs.jsonl` + metadata into ``out_dir``.
    Returns a summary dict. No network, no pixels."""
    results_dir = Path(results_dir)
    manifest_path = Path(manifest_path) if manifest_path else results_dir / "filtered_manifest.json"
    out_dir = Path(out_dir) if out_dir else results_dir / "segment_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)
    results = _load_results(results_dir)

    rows = build_segment_rows(manifest, results)
    df = pd.DataFrame(rows)
    pq = out_dir / "segment_rows.parquet"
    df.to_parquet(pq, index=False)

    qa_path = out_dir / "qa_pairs.jsonl"
    n_pairs = 0
    with open(qa_path, "w") as f:
        for row in rows:
            for pair in render_qa_pairs(row):
                f.write(json.dumps(pair) + "\n")
                n_pairs += 1

    meta = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_git_sha": git_sha,
        "generated_unix": int(time.time()),
        "layers_joined": sorted(results.keys()),
        "n_rows": len(rows),
        "n_segments": int(df[["video_id", "task_id", "segment_index"]].drop_duplicates().shape[0]) if len(df) else 0,
        "n_clips": int(df["video_id"].nunique()) if len(df) else 0,
        "n_qa_pairs": n_pairs,
    }
    with open(out_dir / "segment_export_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return {**meta, "parquet": str(pq), "qa_pairs": str(qa_path), "out_dir": str(out_dir)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Unified per-segment export + QA pairs (docs/06 Issue 2).")
    ap.add_argument("results_dir", help="Dir holding filtered_manifest.json + 03*_result.json")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--git-sha", default="unknown")
    a = ap.parse_args()
    s = build_segment_dataset(a.results_dir, a.manifest, a.out_dir, a.git_sha)
    print(json.dumps(s, indent=2))
