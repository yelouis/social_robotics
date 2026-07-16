"""SRB v0 — Track-B rendering (docs/07 §1.3, §5.2).

Re-emits every Track-A item with the serialized 03x observation block in
place of the clip. This is the ONE sanctioned use of engine output inside an
item: it is the QUESTION (with confidences, including wrong/low-confidence
channels — that honesty is part of what Track B measures). Track B carries
no pixels → publishable for every ring including Ego4D (docs/04 rule).

Usage: python bench/render_track_b.py
"""
import json

from srb_common import BENCH_DATA, read_jsonl, write_jsonl


def _fmt(v, nd=2):
    return "?" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def observation_block(moment) -> str:
    w = moment["window_sec"]
    lines = [f"[social signals, window {w[0]:.1f}-{w[1]:.1f} s]"]
    pf = (moment.get("engine_prefill") or {}).get("per_person", [])
    ambient_done = False
    wearer_line = None
    for p in pf:
        pid = p.get("person_id")
        if pid is not None:
            seg = [f"bystander P{pid}:"]
            if p.get("attention_mean_score") is not None:
                seg.append(f"attention {_fmt(p['attention_mean_score'])} "
                           f"({p.get('attention_n_trace') or 0} samples)")
            g = p.get("gesture_detected")
            if g and g != "none":
                seg.append(f"head gesture: {g.replace('_', ' ')} "
                           f"(conf {_fmt(p.get('gesture_confidence'))})")
            act = p.get("proxemic_action")
            if act and act != "Neutral":
                seg.append(f"proxemics: {act.replace('_', ' ').lower()} "
                           f"vector {_fmt(p.get('proxemic_vector'))} "
                           f"(conf {_fmt(p.get('proxemic_confidence'))})")
            if p.get("motor_resonance_detected"):
                seg.append(f"sympathetic flinch (empathy {_fmt(p.get('empathy_scalar'))})")
            if p.get("emotion_person_score") is not None:
                seg.append(f"emotion appropriateness {_fmt(p['emotion_person_score'])} (signed)")
            if len(seg) > 1:
                lines.append("  " + " ; ".join(seg))
        if not ambient_done and p.get("audio_present"):
            lines.append(f"  ambient prosody: valence {_fmt(p.get('prosody_scalar'))} "
                         f"tone {p.get('prosody_tone')} (scope: ambient, not speaker-attributed)")
            ambient_done = True
        if wearer_line is None and p.get("wearer_egomotion_mean") is not None:
            wearer_line = (f"  wearer: {int(p.get('wearer_n_hand_detections') or 0)} hand detections "
                           f"in action span; egomotion mean {_fmt(p['wearer_egomotion_mean'])} "
                           f"/ peak {_fmt(p.get('wearer_egomotion_peak'))}")
    if wearer_line:
        lines.append(wearer_line)
    if len(lines) == 1:
        lines.append("  (no per-bystander signals measured in this window)")
    return "\n".join(lines)


def main():
    moments = {m["moment_id"]: m for m in read_jsonl(BENCH_DATA / "candidate_moments.jsonl")}
    out_dir = BENCH_DATA / "items"
    n = 0
    for fam in ("f1", "f2", "f3"):
        src = out_dir / f"items_{fam}_track_a.jsonl"
        if not src.exists():
            continue
        items_b = []
        for it in read_jsonl(src):
            b = dict(it)
            b["item_id"] = it["item_id"].replace(f"-{fam}-", f"-{fam}b-")
            b["track"] = "B"
            mids = it.get("moment_ids") or [it["moment_id"]]
            blocks = [observation_block(moments[m]) for m in mids if m in moments]
            if fam == "f3":
                b["observation"] = "\n--- clip 1 ---\n" + blocks[0] + \
                                   "\n--- clip 2 ---\n" + (blocks[1] if len(blocks) > 1 else "")
            else:
                b["observation"] = blocks[0] if blocks else ""
            b.pop("clip", None)
            b.pop("clips", None)
            items_b.append(b)
        write_jsonl(out_dir / f"items_{fam}_track_b.jsonl", items_b)
        # sealed gold for track B = same answers, remapped ids
        gold = read_jsonl(out_dir / f"gold_{fam}.jsonl")
        gold_b = [dict(g, item_id=g["item_id"].replace(f"-{fam}-", f"-{fam}b-")) for g in gold]
        write_jsonl(out_dir / f"gold_{fam}b.jsonl", gold_b)
        n += len(items_b)
    print(f"[track_b] rendered {n} Track-B items")


if __name__ == "__main__":
    main()
