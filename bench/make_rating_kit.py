"""SRB v0 — cut the rating kit per candidate moment (docs/07 §4.2).

Per moment: Clip S ([t−4 s, t+3 s], WITH audio, ≤720p), Clip S+ ([t+3 s,
t+12 s]), and an 8-frame contact strip across S. Kits contain Ego4D source
pixels → INTERNAL-ONLY (never publish; same rule as Node-05 renders).

Also emits:
  kits/kit_manifest.csv       — one row per moment (paths, hint)
  ratings_template.csv        — the Stage-A/B form the rater fills (§4.3/4.4)

Remote-runnable by a future trusted rater against their own Ego4D download:
only ffmpeg + the candidate file are required (docs/06 Issue-4 selection).

Usage: python bench/make_rating_kit.py [--limit N] [--only-retest retest_ids.json]
"""
import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

from srb_common import (BENCH_DATA, CLIP_S_PRE_SEC, CLIP_S_POST_SEC,
                        CLIP_SPLUS_END_SEC, STRIP_FRAMES, RATING_COLUMNS,
                        read_jsonl)

FFMPEG = "ffmpeg"


def cut(video, start, end, out, height=720):
    if out.exists():
        return True
    cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-ss", f"{max(0, start):.2f}",
           "-to", f"{end:.2f}", "-i", str(video),
           "-vf", f"scale=-2:'min({height},ih)'", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-c:a", "aac", "-y", str(out)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def strip(video, start, end, out, n=STRIP_FRAMES):
    if out.exists():
        return True
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    frames = []
    for i in range(n):
        t = start + (end - start) * i / max(1, n - 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000)
        ok, fr = cap.read()
        if ok:
            fr = cv2.resize(fr, (320, 180))
            cv2.putText(fr, f"{t:.1f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            frames.append(fr)
    cap.release()
    if not frames:
        return False
    cv2.imwrite(str(out), np.hstack(frames))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-retest", default=None,
                    help="JSON list of moment_ids: build a RETEST kit (fresh shuffled "
                         "order, separate template) for the §4.5 washout re-rate.")
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()

    moments = read_jsonl(BENCH_DATA / "candidate_moments.jsonl")
    with open(BENCH_DATA / "heldout_manifest.json") as f:
        vids = {e.get("id", e.get("video_id")): e.get("video_path", e.get("file_path"))
                for e in json.load(f)}

    retest = None
    if a.only_retest:
        retest = set(json.loads(Path(a.only_retest).read_text()))
        moments = [m for m in moments if m["moment_id"] in retest]
        random.Random(a.seed).shuffle(moments)         # fresh order, blind re-rate
    if a.limit:
        moments = moments[:a.limit]

    kits = BENCH_DATA / ("kits_retest" if retest else "kits")
    kits.mkdir(parents=True, exist_ok=True)
    rows, fails = [], 0
    for m in moments:
        video = vids.get(m["clip_id"])
        if not video:
            fails += 1
            continue
        t = m["t_climax_sec"]
        d = kits / m["moment_id"]
        d.mkdir(exist_ok=True)
        ok_s = cut(video, t - CLIP_S_PRE_SEC, t + CLIP_S_POST_SEC, d / "clip_s.mp4")
        ok_p = cut(video, t + CLIP_S_POST_SEC, t + CLIP_SPLUS_END_SEC, d / "clip_splus.mp4")
        ok_t = strip(video, t - CLIP_S_PRE_SEC, t + CLIP_S_POST_SEC, d / "strip.jpg")
        if not (ok_s and ok_t):
            fails += 1
            continue
        rows.append({
            "moment_id": m["moment_id"], "clip_id": m["clip_id"],
            "t_climax_sec": t, "has_splus": ok_p,
            "kit_dir": str(d.relative_to(BENCH_DATA)),
            # task_label_hint shown grayed/machine-generated (§4.2);
            # engine_prefill is deliberately ABSENT from kits (§4.6).
            "task_label_hint": m.get("task_label_hint") or "",
        })

    with open(kits / "kit_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmpl = BENCH_DATA / ("ratings_maintainer_retest.csv" if retest else "ratings_maintainer.csv")
    if not tmpl.exists():
        with open(tmpl, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RATING_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({"moment_id": r["moment_id"]})
    print(f"[kits] {len(rows)} kits ({fails} failures) -> {kits}")
    print(f"[kits] rating template -> {tmpl}")


if __name__ == "__main__":
    main()
