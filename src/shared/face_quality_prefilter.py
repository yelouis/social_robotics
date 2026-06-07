"""Shared bystander-face-quality pre-filter.

Layer 03b is intrinsically **low-yield on egocentric footage** (03b Resolved
Issue #7: only ~2/50 clips produce a confident emotion score, because most
bystander crops are too distant/blurry for a face model to resolve), while the
climax optical-flow pass is the **dominant per-clip cost** that every Layer 03
pays first. Paying climax on the ~84-96 % of clips that will score nothing is
wasted work.

This module is the cheap gate that removes that waste. It decodes only the
**sparse bystander-detection frames already recorded in the manifest** (not the
whole clip), runs BlazeFace on each bystander crop, and records the best face
size (px) + confidence per clip under a `bystander_face_quality` manifest field.
A configurable threshold then lets the climax / Layer-03 runner skip clips with
no close, resolvable frontal face *before* paying climax. Measured at
~0.9 s/clip (~15 min for the 1,000-clip reservoir); validated on the June 5
50-clip 03b run to keep both clips that actually scored (0 false negatives)
while skipping 84 % -> ~6.2x climax saving.

Productionizes `scratch/face_quality_prefilter.py` per 03b Unresolved Issue 1
(Option A). Reusable by every face-based Layer 03 module (03b/03c/...).

CLI (spawn-safe standalone pre-pass):
    python -m shared.face_quality_prefilter <filtered_manifest.json>
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2

# MediaPipe BlazeFace (reuses 03a/03b's model). Guarded so a missing optional
# dependency degrades to a no-op (annotation skipped, gate fails OPEN) rather
# than crashing a Layer-03 run.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover - exercised only when mediapipe absent
    mp = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FACE_MODEL = _PROJECT_ROOT / "models" / "mediapipe" / "blaze_face_short_range.tflite"

# Cap on decoded frames per clip — enough to know whether a good face EVER
# appears, while keeping the pass cheap (~0.9 s/clip).
MAX_FRAMES_PER_CLIP = 24
# Confidence at which BlazeFace itself reports a detection while sampling.
DETECT_CONF = 0.5

# Gate thresholds (env-overridable). Defaults are the values validated on the
# June 5 50-clip run: face >= 120 px tall, conf >= 0.8, >= 3 face-frames keeps
# both clips that scored (0 false negatives) and skips 84 % of the rest.
GATE_ENABLED = os.getenv("SR_FACE_QUALITY_GATE", "1").lower() in ("1", "true", "yes")
MIN_FACE_PX = int(os.getenv("SR_FACE_QUALITY_MIN_PX", "120"))
MIN_FACE_CONF = float(os.getenv("SR_FACE_QUALITY_MIN_CONF", "0.8"))
MIN_FACE_FRAMES = int(os.getenv("SR_FACE_QUALITY_MIN_FRAMES", "3"))

FIELD = "bystander_face_quality"


def _make_detector():
    opts = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=DETECT_CONF,
    )
    return mp_vision.FaceDetector.create_from_options(opts)


def compute_face_quality_for_entry(detector, entry: dict) -> dict:
    """Decode the sparse bystander-detection frames for one manifest entry and
    return the best BlazeFace face size/confidence seen.

    Returns ``{best_face_px, best_face_conf, n_face_frames, n_checked}``. A clip
    with no bystander detections (or one that fails to open) returns an all-zero
    record, which the threshold below treats as a fail (skip).
    """
    # Gather (t, bbox) pairs across all bystanders, subsampled to MAX_FRAMES.
    pairs = []
    for b in entry.get("bystander_detections", []):
        for t, bb in zip(b.get("timestamps_sec", []), b.get("bounding_boxes", [])):
            pairs.append((t, bb))
    if not pairs:
        return {"best_face_px": 0, "best_face_conf": 0.0, "n_face_frames": 0, "n_checked": 0}
    if len(pairs) > MAX_FRAMES_PER_CLIP:
        step = len(pairs) / MAX_FRAMES_PER_CLIP
        pairs = [pairs[int(i * step)] for i in range(MAX_FRAMES_PER_CLIP)]

    video_path = entry.get("video_path")
    if not video_path or not Path(video_path).exists():
        return {"best_face_px": 0, "best_face_conf": 0.0, "n_face_frames": 0, "n_checked": 0}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return {"best_face_px": 0, "best_face_conf": 0.0, "n_face_frames": 0, "n_checked": 0}
    fps = cap.get(cv2.CAP_PROP_FPS) or entry.get("fps") or 30.0

    best_px, best_conf, n_face = 0, 0.0, 0
    for t, bb in pairs:
        # Random seek is the RIGHT choice here (unlike climax/03a, which decode
        # consecutive frames and must use sequential grab()): the samples are a
        # handful of widely-spaced timestamps across the whole clip, so seeking
        # to each is far cheaper than decoding every frame in between. Exact
        # frame accuracy is irrelevant — we only need a roughly-right frame to
        # see whether a resolvable face is present.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bb]
        crop = frame[max(0, y1 - 20):min(h, y2 + 20), max(0, x1 - 20):min(w, x2 + 20)]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        det = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if det.detections:
            n_face += 1
            for d in det.detections:
                best_px = max(best_px, d.bounding_box.height)
                best_conf = max(best_conf, d.categories[0].score)
    cap.release()
    return {
        "best_face_px": int(best_px),
        "best_face_conf": round(float(best_conf), 3),
        "n_face_frames": n_face,
        "n_checked": len(pairs),
    }


def passes_face_quality(
    entry: dict,
    min_px: int = MIN_FACE_PX,
    min_conf: float = MIN_FACE_CONF,
    min_frames: int = MIN_FACE_FRAMES,
    enabled: Optional[bool] = None,
) -> bool:
    """Gate predicate: ``True`` if the entry's recorded face quality clears the
    threshold (i.e. it is worth paying climax + emotion on).

    Fail-open semantics so the gate never *silently* discards work:
    - gate disabled -> always ``True``;
    - entry has no ``bystander_face_quality`` field (never pre-scored) -> ``True``
      (we did not check, so we do not drop it);
    - field present -> apply the px/conf/frame-count threshold. A present-but-
      zero record (checked, no resolvable face) therefore correctly fails.
    """
    if enabled is None:
        enabled = GATE_ENABLED
    if not enabled:
        return True
    fq = entry.get(FIELD)
    if not fq:
        return True  # not pre-scored -> fail open
    return (
        fq.get("best_face_px", 0) >= min_px
        and fq.get("best_face_conf", 0.0) >= min_conf
        and fq.get("n_face_frames", 0) >= min_frames
    )


def populate_face_quality_for_manifest(
    manifest_path,
    force: bool = False,
    save_every: int = 25,
) -> int:
    """Annotate every entry in ``manifest_path`` with a ``bystander_face_quality``
    record, writing back to the same path. Returns the number of entries scored.

    Idempotent — entries already carrying the field are skipped unless
    ``force=True``, so a Layer-03 run can call this unconditionally and only pay
    for clips not yet scored. The manifest is rewritten every ``save_every``
    entries (a save point, so a crash mid-pass does not lose progress) and once
    at the end. Defensive: if mediapipe is unavailable the pass is a logged
    no-op (returns 0) and the downstream gate fails open.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return 0
    if mp is None or not FACE_MODEL.exists():
        print("[face_quality] mediapipe/BlazeFace unavailable — skipping pre-filter "
              "(gate will fail open).")
        return 0

    with open(manifest_path, "r") as f:
        entries = json.load(f)

    todo = [i for i, e in enumerate(entries)
            if force or not e.get(FIELD)]
    if not todo:
        return 0

    detector = _make_detector()
    scored = 0
    t0 = time.time()
    for n, i in enumerate(todo, 1):
        entry = entries[i]
        entry[FIELD] = compute_face_quality_for_entry(detector, entry)
        scored += 1
        if n % save_every == 0 or n == len(todo):
            with open(manifest_path, "w") as f:
                json.dump(entries, f, indent=4)
            vid = entry.get("video_id", entry.get("id", "?"))
            print(f"[face_quality] {n}/{len(todo)} {str(vid)[:8]} {entry[FIELD]} "
                  f"({(time.time() - t0) / n:.1f}s/clip)", flush=True)
    print(f"[face_quality] scored {scored} entries in {time.time() - t0:.0f}s "
          f"-> {manifest_path}", flush=True)
    return scored


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Annotate a filtered manifest with bystander_face_quality, "
                    "so the climax / Layer-03 runner can skip low-yield clips.")
    ap.add_argument("manifest")
    ap.add_argument("--force", action="store_true",
                    help="Re-score entries that already have the field.")
    a = ap.parse_args()
    _t0 = time.time()
    n = populate_face_quality_for_manifest(a.manifest, force=a.force)
    # Report how many of the (now-annotated) entries the default gate keeps.
    with open(a.manifest) as f:
        _entries = json.load(f)
    _kept = sum(1 for e in _entries if passes_face_quality(e))
    print(f"[face_quality] done: scored {n} in {time.time() - _t0:.0f}s; "
          f"gate keeps {_kept}/{len(_entries)} "
          f"(px>={MIN_FACE_PX}, conf>={MIN_FACE_CONF}, frames>={MIN_FACE_FRAMES}).")
