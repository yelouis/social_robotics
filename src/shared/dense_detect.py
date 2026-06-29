"""
Windowed dense bystander detection (shared).

Node-02 samples bystander boxes at ~1 frame / 3 s — far too sparse for a box to
*follow* a moving person, and sparse enough that several Layer-03 metrics fall
back to interpolating/carrying a stale box (03f crops to an interpolated bbox;
its camera-motion gate then forces velocity to 0 when < 2 genuine detections land
in the window).

This module runs **plain YOLO + ByteTrack on the actual frames inside a few short
time windows** (e.g. the task reaction windows), then **matches the dense tracks
back to the existing manifest `person_id`s** by IoU. No VLM, no keep/drop logic —
the manifest already tells us who the verified bystanders are; we only need dense
box *positions* for them.

Used two ways:
  * the Layer-05 visualizer (`--dense-boxes`) so boxes track the subject, and
  * an A/B harness that feeds densified manifests to Layer-03 to measure impact.

It is deliberately NOT wired into Node-02 — see docs/05 for the "densify at the
point of use" rationale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from src.models_config import get_model
except ImportError:  # pragma: no cover - import shim mirrors the layer modules
    from models_config import get_model


def iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pick_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class _DenseDetector:
    """Lazily-loaded YOLO person tracker (reused across clips in one process)."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        from ultralytics import YOLO
        if model_path is None:
            weights = get_model("social_presence_pose")  # yolov8n-pose.pt
            base = Path(__file__).resolve().parent.parent.parent
            cand = base / weights
            model_path = str(cand) if cand.exists() else weights
        self.model = YOLO(model_path)
        self.device = device or _pick_device()

    def track_window(self, cap, fps, start_sec, end_sec, fps_target, conf):
        """Track persons across one [start_sec, end_sec] window of an open capture.

        Returns {track_id: [(t, [x1,y1,x2,y2], conf), ...]} (sorted by t).
        ByteTrack is reset at the window start so IDs are local to the window.
        """
        stride = max(1, int(round(fps / fps_target)))
        start_frame = max(0, int(start_sec * fps))
        end_frame = int(end_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        tracks: dict[int, list] = {}
        fidx = start_frame
        first = True
        while fidx <= end_frame:
            if (fidx - start_frame) % stride != 0:
                if not cap.grab():
                    break
                fidx += 1
                continue
            ret, frame = cap.read()
            if not ret:
                break
            t = fidx / fps
            fidx += 1
            # persist=False on the first frame resets ByteTrack for this window.
            res = self.model.track(
                frame, persist=not first, classes=[0], conf=conf,
                verbose=False, tracker="bytetrack.yaml", device=self.device,
            )
            first = False
            r = res[0]
            if r.boxes is None or r.boxes.id is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for box, tid, c in zip(xyxy, ids, confs):
                tracks.setdefault(int(tid), []).append(
                    (t, [float(box[0]), float(box[1]), float(box[2]), float(box[3])], float(c))
                )
        for tid in tracks:
            tracks[tid].sort(key=lambda s: s[0])
        return tracks


def _match_tracks_to_person(tracks, person_boxes_in_window, iou_thresh=0.3):
    """Pick the dense track that best matches one manifest person's window boxes.

    `person_boxes_in_window` = [(t_manifest, box_manifest), ...]. For each manifest
    sample we find the track box nearest in time and sum IoU; the highest-scoring
    track above threshold wins. Returns (track_id | None, score).
    """
    best_tid, best_score = None, 0.0
    for tid, samples in tracks.items():
        ts = [s[0] for s in samples]
        score = 0.0
        for tm, bm in person_boxes_in_window:
            j = min(range(len(ts)), key=lambda k: abs(ts[k] - tm))
            score += iou(samples[j][1], bm)
        if score > best_score:
            best_tid, best_score = tid, score
    # require the average per-sample IoU to clear the threshold
    if best_tid is not None and person_boxes_in_window:
        if best_score / len(person_boxes_in_window) >= iou_thresh:
            return best_tid, best_score
    return None, 0.0


def densify_bystander_detections(
    video_path, bystanders, windows, *,
    fps_target=10.0, conf=0.35, pad_sec=4.0, detector: Optional[_DenseDetector] = None,
):
    """Return a NEW bystander_detections list with dense boxes inside `windows`.

    For each window we run dense YOLO+ByteTrack, match each tracked person to a
    manifest `person_id` (IoU vote at that person's manifest timestamps), and
    splice the dense samples into that person's detection arrays *inside the
    window* while keeping their original sparse samples *outside* every window.
    Persons with no confident match keep their sparse boxes unchanged (honest).

    Returns (new_bystanders, stats) where stats summarizes what was densified.
    """
    detector = detector or _DenseDetector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return bystanders, {"error": "open_failed"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # padded, merged windows
    pw = []
    for w in windows:
        if w and len(w) == 2:
            pw.append((max(0.0, float(w[0]) - pad_sec), float(w[1]) + pad_sec))
    pw.sort()
    merged = []
    for s, e in pw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # dense tracks per window (namespaced by window index)
    window_tracks = []
    try:
        for s, e in merged:
            window_tracks.append((s, e, detector.track_window(cap, fps, s, e, fps_target, conf)))
    finally:
        cap.release()

    def _in_any_window(t):
        return any(s <= t <= e for s, e, _ in window_tracks)

    new_bystanders = []
    stats = {"persons": 0, "densified_persons": 0, "added_samples": 0}
    for b in bystanders:
        stats["persons"] += 1
        pid = b.get("person_id")
        ts = list(b.get("timestamps_sec", []) or [])
        bxs = list(b.get("bounding_boxes", []) or [])
        cfs = list(b.get("detection_confidence", []) or [])
        # keep this person's sparse samples that fall OUTSIDE all windows
        kept = [(ts[i], bxs[i], cfs[i] if i < len(cfs) else 1.0)
                for i in range(min(len(ts), len(bxs))) if not _in_any_window(ts[i])]

        dense_added = []
        for s, e, tracks in window_tracks:
            person_window_boxes = [(ts[i], bxs[i]) for i in range(min(len(ts), len(bxs)))
                                   if s <= ts[i] <= e]
            if not person_window_boxes:
                # no manifest anchor for this person in this window -> can't match safely
                continue
            tid, _score = _match_tracks_to_person(tracks, person_window_boxes)
            if tid is None:
                continue
            for (t, box, c) in tracks[tid]:
                dense_added.append((t, [int(round(v)) for v in box], c))

        if dense_added:
            stats["densified_persons"] += 1
            stats["added_samples"] += len(dense_added)
            allsamples = kept + dense_added
        else:
            # fall back to the original (full) sparse track
            allsamples = [(ts[i], bxs[i], cfs[i] if i < len(cfs) else 1.0)
                          for i in range(min(len(ts), len(bxs)))]
        allsamples.sort(key=lambda s: s[0])

        nb = dict(b)
        nb["timestamps_sec"] = [round(s[0], 3) for s in allsamples]
        nb["bounding_boxes"] = [s[1] for s in allsamples]
        nb["detection_confidence"] = [s[2] for s in allsamples]
        new_bystanders.append(nb)

    return new_bystanders, stats


def _entry_windows(entry, pad_sec=0.0):
    """Collect the reaction windows from an entry's identified_tasks."""
    out = []
    for t in entry.get("identified_tasks", []) or []:
        tm = t.get("task_temporal_metadata", {}) or {}
        rw = tm.get("task_reaction_window_sec")
        if rw and len(rw) == 2:
            out.append([float(rw[0]) - pad_sec, float(rw[1]) + pad_sec])
    return out


def densify_manifest_entry(entry, *, fps_target=10.0, pad_sec=4.0, conf=0.35,
                           detector: Optional[_DenseDetector] = None):
    """Return a copy of a manifest entry with dense bystander boxes in its task
    reaction windows. Non-destructive; leaves everything else untouched."""
    video_path = entry.get("video_path")
    bystanders = entry.get("bystander_detections", []) or []
    windows = _entry_windows(entry)
    if not video_path or not Path(video_path).exists() or not bystanders or not windows:
        return entry, {"skipped": "missing_inputs"}
    new_b, stats = densify_bystander_detections(
        video_path, bystanders, windows,
        fps_target=fps_target, conf=conf, pad_sec=pad_sec, detector=detector,
    )
    out = dict(entry)
    out["bystander_detections"] = new_b
    return out, stats
