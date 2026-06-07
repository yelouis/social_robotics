"""Validate sequential-decode climax: recompute on clips that already have
old (cap.set) cached climax, compare timestamps + measure speedup."""
import json, time, copy, glob, sys
from pathlib import Path
sys.path.insert(0, "src")
import cv2
from shared.climax_extraction import compute_task_climax_for_video

man = json.load(open(glob.glob("e2e_reports/*layer03b/manifest_03b_10.json")[0]))
print("clip       old_climax  new_climax  diff_s  new_time", flush=True)
tnew = 0.0
maxdiff = 0.0
for e in man[:6]:
    told_meta = e["identified_tasks"][0].get("task_temporal_metadata")
    if not told_meta:
        continue
    cap = cv2.VideoCapture(e["video_path"]); fps = cap.get(cv2.CAP_PROP_FPS)
    tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); dur = e["duration_sec"]
    tasks_new = copy.deepcopy(e["identified_tasks"])
    for t in tasks_new:
        t["task_temporal_metadata"] = {}
    t0 = time.time()
    compute_task_climax_for_video(cap, fps, tf, tasks_new, dur, skip_vlm=True)
    dt = time.time() - t0
    cap.release()
    oc = told_meta.get("task_climax_sec")
    nc = tasks_new[0]["task_temporal_metadata"].get("task_climax_sec")
    d = abs((oc or 0) - (nc or 0)); maxdiff = max(maxdiff, d); tnew += dt
    print(f"  {e['video_id'][:8]}  {oc:>9}  {nc:>9}  {d:>5.1f}  {dt:.0f}s", flush=True)
print(f"\nNEW total 6 clips: {tnew:.0f}s | old was ~99s/clip (~594s for 6) -> "
      f"speedup ~{594/max(1,tnew):.1f}x | max climax diff {maxdiff:.1f}s", flush=True)
print("VALIDATE_DONE", flush=True)
