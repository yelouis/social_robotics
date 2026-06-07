"""Render a gaze-overlay smell-test frame for a 03a result.

For a given clip + person, pick the trace point matching a target score (max or
min), extract that frame, draw the bystander bbox + the gaze direction arrow
(from trace pitch/yaw, same convention as the pipeline) + the score/target label.
Lets us visually judge: does 03a's score match where the bystander is looking?
"""
import json, math, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
RDIR = ROOT / "e2e_reports" / "2026_06_02_layer03a"
import os as _os
_RESFILE = _os.getenv("SR_03A_RESULT", "03a_attention_result_10.json")
res = {r["video_id"]: r for r in json.load(open(RDIR / _RESFILE))}
man = {e["video_id"]: e for e in json.load(open(RDIR / "manifest_10.json"))}

def nearest_bbox(bystander, t):
    ts = bystander["timestamps_sec"]; bb = bystander["bounding_boxes"]
    i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
    return bb[i], abs(ts[i] - t)

def render(vid, which, out_name, person_idx=None):
    r = res[vid]; e = man[vid]
    # choose the person with the most trace points (most-tracked), or given idx
    persons = r["per_person"]
    p = persons[person_idx] if person_idx is not None else max(persons, key=lambda pp: len(pp["attention_trace"]))
    pid = p["person_id"]
    trace = [t for t in p["attention_trace"]]
    if not trace: print(f"{vid[:8]}: empty trace"); return
    pt = (max if which == "max" else min)(trace, key=lambda x: x["score"])
    t = pt["t"]
    bystander = next(b for b in e["bystander_detections"] if b["person_id"] == pid)
    bbox, dt = nearest_bbox(bystander, t)
    cap = cv2.VideoCapture(e["video_path"]); fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, frame = cap.read(); cap.release()
    if not ok: print(f"{vid[:8]}: could not read frame at t={t}"); return
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # gaze arrow (pipeline convention: v_look_x=-cos(yaw)sin(pitch), v_look_y=-sin(yaw))
    yaw, pitch = pt["yaw_rad"], pt["pitch_rad"]
    vx = -math.cos(yaw) * math.sin(pitch); vy = -math.sin(yaw)
    L = max(60, (x2 - x1))
    ex, ey = int(cx + vx * L), int(cy + vy * L)
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0, 0, 255), 4, tipLength=0.3)
    label = f"score={pt['score']:.2f} target={pt['target']} t={t:.1f}s pitch={pitch:.2f} yaw={yaw:.2f}"
    cv2.rectangle(frame, (0, 0), (min(w, 760), 34), (0, 0, 0), -1)
    cv2.putText(frame, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    dest = RDIR / "frames" / out_name
    cv2.imwrite(str(dest), frame)
    print(f"  {vid[:8]} [{which}] t={t:.1f}s score={pt['score']:.2f} target={pt['target']} "
          f"bbox_dt={dt:.2f}s -> {dest.name}")

if __name__ == "__main__":
    # vid, which, outname[, person_idx]
    a = sys.argv[1:]
    render(next(v for v in man if v.startswith(a[0])), a[1], a[2],
           int(a[3]) if len(a) > 3 else None)
