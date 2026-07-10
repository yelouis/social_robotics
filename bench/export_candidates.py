"""SRB v0 — export `candidate_moments.jsonl` (docs/07 §0 interface).

Thin shim over the 02b-annotated held-out manifest (segments via
`shared.climax_extraction.expand_task_segments` — the one sanctioned engine
import) plus the segment-join ARTIFACT (`segment_dataset/segment_rows.parquet`,
read as a file, never imported) for `engine_prefill`.

Only clips stamped `test_heldout` in bench/splits_ego4d.json are exported —
a published or wearer-overlapping clip in the held-out manifest is a
selection bug and raises.

Usage: python bench/export_candidates.py
Output: bench_v0/candidate_moments.jsonl
"""
import json
import sys

import pandas as pd

from srb_common import (BENCH_DATA, REPO_ROOT, SPLITS_FILE, moment_id,
                        write_jsonl)

sys.path.insert(0, str(REPO_ROOT / "src"))
from shared.climax_extraction import expand_task_segments  # noqa: E402  (sanctioned)

PREFILL_COLS = [
    "attention_mean_score", "attention_n_trace", "attention_n_head_pose",
    "gesture_detected", "gesture_confidence",
    "proxemic_vector", "proxemic_action", "proxemic_confidence",
    "prosody_scalar", "prosody_tone", "audio_present",
    "motor_resonance_detected", "empathy_scalar", "mirroring_detected",
    "emotion_task_score", "emotion_person_score",
    "ego_kinetic_chaos_score", "segment_face_px",
    "wearer_n_hand_detections", "wearer_egomotion_mean", "wearer_egomotion_peak",
]


def main():
    with open(BENCH_DATA / "heldout_manifest.json") as f:
        manifest = json.load(f)
    splits = json.loads(SPLITS_FILE.read_text())["splits"]

    seg_rows = pd.read_parquet(BENCH_DATA / "segment_dataset" / "segment_rows.parquet")
    by_seg = {k: g for k, g in seg_rows.groupby(["video_id", "task_id", "segment_index"])}

    moments = []
    for entry in manifest:
        vid = entry.get("id", entry.get("video_id"))
        split = splits.get(vid)
        if split != "test_heldout":
            raise SystemExit(f"[export] {vid} has split={split!r} — held-out manifest "
                             "must contain only test_heldout clips (selection bug).")
        for pseudo in expand_task_segments(entry.get("identified_tasks", [])):
            meta = pseudo["task_temporal_metadata"]
            t = meta["task_climax_sec"]
            w = meta["task_reaction_window_sec"]
            g = by_seg.get((vid, pseudo.get("task_id", "unknown"), pseudo["segment_index"]))
            prefill = {}
            if g is not None:
                per_person = []
                for _, r in g.iterrows():
                    d = {c: (None if pd.isna(r[c]) else
                             (r[c].item() if hasattr(r[c], "item") else r[c]))
                         for c in PREFILL_COLS if c in g.columns}
                    d["person_id"] = None if pd.isna(r["person_id"]) else int(r["person_id"])
                    per_person.append(d)
                prefill = {"per_person": per_person}
            moments.append({
                "moment_id": moment_id("ego4d", vid, t),
                "corpus": "ego4d",
                "clip_id": vid,
                "t_climax_sec": round(float(t), 2),
                "window_sec": [round(float(w[0]), 2), round(float(w[1]), 2)],
                "source": "engine",
                "is_control": bool(pseudo.get("is_control", False)),
                "control_type": pseudo.get("control_type"),
                "engine_prefill": prefill,                 # HIDDEN from test raters
                "task_label_hint": (meta.get("segment_action_caption")
                                    or pseudo.get("task_label")),
            })

    out = BENCH_DATA / "candidate_moments.jsonl"
    write_jsonl(out, moments)
    n_ctrl = sum(1 for m in moments if m["is_control"])
    print(f"[export] {len(moments)} candidate moments ({n_ctrl} engine controls) "
          f"from {len(manifest)} clips -> {out}")


if __name__ == "__main__":
    main()
