"""03b spot-check: for a clip, re-sample the bystander crop across the reaction
window (same bbox + pad the pipeline uses), run the same ONNX HSEmotion, and
overlay the emotion label + magnitude on the frame so we can judge whether the
detected emotion — and thus the success/failure direction — matches the face.

Usage: PYTHONPATH=src python scratch/spotcheck_03b.py <video_id_prefix> <out_prefix>
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

ROOT = Path(__file__).resolve().parent.parent
RDIR = ROOT / "e2e_reports" / "2026_06_04_layer03b"
man = {e["video_id"]: e for e in json.load(open(RDIR / "manifest_03b_10.json"))}
res = {r["video_id"]: r for r in json.load(open(RDIR / "03b_result_10.json"))}
rec = HSEmotionRecognizer(model_name="enet_b2_8")

def nearest_bbox(b, t):
    ts, bb = b["timestamps_sec"], b["bounding_boxes"]
    i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
    return bb[i], abs(ts[i] - t)

def run(vid_prefix, out_prefix):
    vid = next(v for v in man if v.startswith(vid_prefix))
    e = man[vid]
    # pick the task with a reaction window + its first analyzed person
    r = res.get(vid)
    task = e["identified_tasks"][0]
    win = task.get("task_temporal_metadata", {}).get("task_reaction_window_sec")
    if not win:
        print(f"{vid[:8]}: no reaction window"); return
    # person to inspect: the one 03b analyzed (from result), else first bystander
    pid = None
    if r and r["tasks_analyzed"] and r["tasks_analyzed"][0]["per_person"]:
        pid = r["tasks_analyzed"][0]["per_person"][0]["person_id"]
        score = r["tasks_analyzed"][0]["task_aggregate_score"]
    else:
        score = None
    bystander = next((b for b in e["bystander_detections"] if b["person_id"] == pid), e["bystander_detections"][0])
    pid = bystander["person_id"]
    cap = cv2.VideoCapture(e["video_path"]); fps = cap.get(cv2.CAP_PROP_FPS)
    s, en = win
    times = list(np.linspace(s, en, 4))
    for i, t in enumerate(times):
        bbox, dt = nearest_bbox(bystander, t)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, frame = cap.read()
        if not ok: continue
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
        px2, py2 = min(w, x2 + 20), min(h, y2 + 20)
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0: continue
        lab, sc = rec.predict_emotions(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), logits=False)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"emotion={lab} mag={float(max(sc)):.2f} t={t:.1f}s win={s:.0f}-{en:.0f} score={score}"
        cv2.rectangle(frame, (0, 0), (min(w, 900), 34), (0, 0, 0), -1)
        cv2.putText(frame, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        dest = RDIR / "frames" / f"{out_prefix}_{i+1}.jpg"
        cv2.imwrite(str(dest), frame)
        print(f"  {vid[:8]} t={t:.1f}s emotion={lab} mag={float(max(sc)):.2f} bbox_dt={dt:.2f}s -> {dest.name}")
    cap.release()

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
