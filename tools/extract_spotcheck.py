"""Flexible spot-check frame extractor for E2E reports.

Usage:
  python extract_spotcheck.py --video PATH --prefix NAME --outdir DIR \
      [--timestamps 10,60,120] [--n 3] [--boxes-from RESULTS.json --video-id ID]

- If --timestamps given, extracts those exact seconds.
- Else if --boxes-from + --video-id, uses that record's bystander timestamps
  and draws the detected bounding boxes on the frame.
- Else auto-samples --n evenly spaced frames across the clip.
Frames are downscaled to <=640px on the long edge for embedding.
"""
import argparse
import json
from pathlib import Path

import cv2


def _load_boxes(results_path, video_id):
    with open(results_path) as f:
        recs = json.load(f)
    rec = next((r for r in recs if r.get("video_id") == video_id), None)
    if not rec:
        return [], []
    ts, boxes = [], []
    for b in rec.get("bystander_detections", []):
        for t, bb in zip(b["timestamps_sec"], b["bounding_boxes"]):
            ts.append(t)
            boxes.append(bb)
    # sort by timestamp, keep co-indexed
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    return [ts[i] for i in order], [boxes[i] for i in order]


def extract(video_path, outdir, prefix, timestamps, boxes=None, max_frames=3):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"FAILED to open {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved = []
    sel = timestamps[:max_frames]
    for i, t in enumerate(sel):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        if boxes is not None and i < len(boxes) and boxes[i]:
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        h, w = frame.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            frame = cv2.resize(frame, (0, 0), fx=s, fy=s)
        out = Path(outdir) / f"{prefix}_{i+1}.jpg"
        cv2.imwrite(str(out), frame)
        saved.append(str(out))
        print(f"  saved {out} @ t={t:.1f}s ({w}x{h})")
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--timestamps", default=None, help="comma-separated seconds")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--boxes-from", default=None)
    ap.add_argument("--video-id", default=None)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    boxes = None

    if args.timestamps:
        ts = [float(x) for x in args.timestamps.split(",")]
    elif args.boxes_from and args.video_id:
        ts, boxes = _load_boxes(args.boxes_from, args.video_id)
        if not ts:
            print(f"No bystander timestamps for {args.video_id}; falling back to even sampling.")
    else:
        ts = []

    if not ts:
        cap = cv2.VideoCapture(str(args.video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        dur = total / fps if fps else 0
        n = args.n
        ts = [dur * (k + 1) / (n + 1) for k in range(n)]

    print(f"Extracting {args.prefix} from {args.video}")
    extract(args.video, args.outdir, args.prefix, ts, boxes=boxes, max_frames=args.n)


if __name__ == "__main__":
    main()
