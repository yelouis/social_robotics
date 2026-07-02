"""Layer 02b — Task Climax / Reaction Segments.

Annotates the Node-02 `filtered_manifest.json` in place: for every
`identified_task` it fills `task_temporal_metadata` with a list of
bystander-anchored `reaction_segments` that downstream Layer 03 pipelines
consume via `shared.climax_extraction.iter_reaction_windows()` /
`expand_task_segments()`.

Position in the pipeline: **after Node 02, before every Layer 03x**. Node 02
emits `task_temporal_metadata = {}` (docs/02 Resolved #8); this layer is the
stage that fills it. Layer 03 pipelines still invoke it lazily through the
back-compat `shared.climax_extraction.populate_climax_for_manifest()` wrapper,
so an un-annotated manifest degrades gracefully — but the supported production
path is to run this layer explicitly first (see CLI below).

### Detector (bbox-kernel peak, June 30 A/B rework)

The original detector maximized global-frame Farneback optical flow inside
each bystander cluster. The June 30 paired A/B eval (8 clips / 24 paired
windows / real 03a->03e runs; see docs/02b_task_climax_layer.md) showed that
peak carries no social information on egocentric footage — it is dominated by
ego-motion and passing objects (a passing car; walking-gait bob), landing
uniform-randomly within the cluster and >2 s from any bystander detection 16 %
of the time. It was also the dominant per-clip cost of every Layer 03 run.

The reworked detector needs **no video decode** for the climax choice:

1. Bystander detections are clustered in time exactly as before
   (`_cluster_timestamps`; gap > 15 s or span > 90 s splits, densest
   `CLIMAX_MAX_SEGMENTS` kept).
2. Within each cluster, the climax is the **detection timestamp maximizing
   Gaussian-kernel-weighted bbox height** — tall boxes = close bystanders =
   resolvable faces, and the kernel rewards *sustained* proximity over a
   single-frame spike. Because the climax IS a detection timestamp, downstream
   window re-anchoring (shared/bystander_window.py) is satisfied by
   construction instead of rescuing a meaningless flow peak.
3. The reaction window **straddles** the climax (`[climax-1 s, climax+3 s]`)
   instead of trailing it — the old "reaction follows wearer-motion peak"
   assumption inverted when the flow peak was the wearer turning *away* — and
   is **shifted, not clipped,** at clip edges (no degenerate `[d, d]` windows).
4. Optional **face verification** (`SR_02B_FACE_VERIFY`, default on): decode
   ONE frame for each of the top-`FACE_VERIFY_CANDIDATES` kernel candidates,
   run BlazeFace on the tallest nearby bystander crop, and pick the candidate
   with the largest resolvable face. This fixes the kernel's one blind spot —
   proximity-aware but orientation-blind (a close bystander facing away) —
   at a cost of ~3 frame decodes per segment (vs thousands for optical flow).
   The best face height is recorded per segment as `segment_face_px`, giving
   downstream layers a free per-segment quality gate.

A/B validation (June 30, vs the flow detector, paired on identical clusters):
2.8x more in-window 03a trace samples, 2.5x more head-pose samples, 03e
measurements 32 -> 35 with direct-window (non-re-anchored) share 10 -> 13,
zero-detection windows 43.5 % -> 0 %, at ~zero compute vs the previously
dominant flow pass.

CLI (spawn-safe; the explicit production pre-pass):
    python -m layer_02b_task_climax.pipeline <filtered_manifest.json> \
        [--workers N] [--no-face-verify] [--force]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FACE_MODEL = _PROJECT_ROOT / "models" / "mediapipe" / "blaze_face_short_range.tflite"

# MediaPipe BlazeFace for segment face verification. Guarded so a missing
# optional dependency degrades to kernel-only selection rather than crashing.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover - exercised only when mediapipe absent
    mp = None

# Action-caption dependencies (docs/06 Issue 1). Both guarded: without them the
# caption pass degrades to a logged no-op and every other 02b output is intact.
try:
    from shared.vlm_client import ollama_chat
except ImportError:
    try:
        from src.shared.vlm_client import ollama_chat
    except ImportError:  # pragma: no cover
        ollama_chat = None
try:
    from models_config import get_model
except ImportError:
    try:
        from src.models_config import get_model
    except ImportError:  # pragma: no cover
        get_model = None

# --- Clustering (unchanged from the flow-era detector; validated by the
# June 23 multi-window rework, docs/02 Resolved #22) ---
CLIMAX_CLUSTER_GAP_SEC = 15.0        # split clusters where bystanders are absent > this
CLIMAX_MAX_CLUSTER_SPAN_SEC = 90.0   # also split clusters longer than this
CLIMAX_MIN_CLUSTER_DETECTIONS = 2    # ignore singleton/spurious clusters
CLIMAX_MAX_SEGMENTS = 10             # cap segments/task (densest kept)
CLIMAX_CHECKPOINT_EVERY = 15         # write the manifest every N entries

# --- Bbox-kernel climax scoring ---
KERNEL_SIGMA_SEC = 1.0     # Gaussian width: rewards sustained presence around t
KERNEL_SUPPORT_SEC = 2.0   # detections beyond this contribute ~nothing
WINDOW_PRE_SEC = 1.0       # window = [climax - PRE, climax + POST]
WINDOW_POST_SEC = 3.0

# --- Face verification (B+; env-overridable) ---
FACE_VERIFY = os.getenv("SR_02B_FACE_VERIFY", "1").lower() in ("1", "true", "yes")
FACE_VERIFY_CANDIDATES = int(os.getenv("SR_02B_FACE_VERIFY_CANDIDATES", "3"))
FACE_DETECT_CONF = 0.5
CROP_PAD_PX = 20           # same bbox pad as shared/face_quality_prefilter.py

# --- Control (negative) segments (docs/06 Issue 3; env-overridable) ---
# Without negatives the dataset is all-positive and a reward model/benchmark
# degenerates to "always predict a social reaction". Two kinds are emitted,
# flagged `is_control: true` and APPENDED after the real segments (so
# segment_index alignment with every layer's result rows is preserved):
#   present_unreactive — inside a bystander cluster but far from its climax
#                        (bystander there, no anchored social moment);
#   no_audience        — from the spans between clusters with no bystander
#                        within the clearance (wearer acting, nobody watching).
# Controls are measured by the 03x layers like any segment — that measurement
# IS the negative label.
CONTROL_SEGMENTS = os.getenv("SR_02B_CONTROL_SEGMENTS", "1").lower() in ("1", "true", "yes")
CONTROL_MAX_PRESENT = 2            # present_unreactive cap per task
CONTROL_MAX_NO_AUDIENCE = 1        # no_audience cap per task
CONTROL_MIN_CLIMAX_SEPARATION_SEC = 10.0   # present_unreactive: min distance from the cluster's climax
CONTROL_MIN_GAP_SEC = 30.0                 # no_audience: min bystander-free gap
CONTROL_CLEARANCE_SEC = 5.0                # no_audience: no detection within this of the window

# --- Per-segment action captions (docs/06 Issue 1; env-overridable) ---
# Captions run only on the EXPLICIT pipeline path (CLI / TaskClimaxPipeline):
# the lazy back-compat wrapper's callers all pass skip_vlm=True, so a Layer 03
# run on an un-annotated manifest never blocks on an absent ollama server.
ACTION_CAPTIONS = os.getenv("SR_02B_ACTION_CAPTIONS", "1").lower() in ("1", "true", "yes")
CAPTION_FRAME_OFFSETS_SEC = (-1.0, 0.5)   # frames sampled around the climax
CAPTION_VLM_TIMEOUT = 180                 # enforced httpx timeout (see shared/vlm_client)
CAPTION_MAX_EDGE_PX = 1024                # downscale bound for the VLM frames
# Sentinel captions the prompt allows the model to answer with instead of
# hallucinating a discrete action (docs/06 Issue 5 brainstorm): 'conversation'
# = wearer is only talking/listening, no discrete physical action;
# 'unclear' = action cannot be determined from the frames.
CAPTION_SENTINELS = ("conversation", "unclear")


def _cluster_timestamps(timestamps: List[float], gap_sec: float,
                        max_span_sec: float) -> List[List[float]]:
    """Group sorted timestamps into `[cluster_start, cluster_end, count]` clusters.
    A new cluster starts when the gap to the previous detection exceeds `gap_sec`
    OR the running span would exceed `max_span_sec` (so a continuously-attended
    long stretch is split into bounded sub-segments rather than one huge window)."""
    if not timestamps:
        return []
    ts = sorted(timestamps)
    clusters: List[List[float]] = []
    cs = ce = ts[0]
    n = 1
    for t in ts[1:]:
        if (t - ce) > gap_sec or (t - cs) > max_span_sec:
            clusters.append([cs, ce, n])
            cs = t
            n = 1
        else:
            n += 1
        ce = t
    clusters.append([cs, ce, n])
    return clusters


def _bystander_timestamps_in(bystander_detections, start_sec: float, end_sec: float) -> List[float]:
    """Flatten every bystander track's detection timestamps that fall inside the
    task's [start, end] range. Kept for API compatibility with the flow-era
    module (tests + potential external callers)."""
    out: List[float] = []
    for track in (bystander_detections or []):
        for t in (track.get('timestamps_sec') or []):
            if start_sec <= t <= end_sec:
                out.append(float(t))
    return out


def _detections_in(bystander_detections, start_sec: float, end_sec: float
                   ) -> List[Tuple[float, float, list]]:
    """Sorted `(t, bbox_height, bbox)` for every detection in [start, end].
    Tracks lacking co-indexed bounding boxes contribute height 0 (they still
    count for clustering, but cannot win the kernel score against a real box)."""
    out: List[Tuple[float, float, list]] = []
    for track in (bystander_detections or []):
        ts = track.get('timestamps_sec') or []
        bbs = track.get('bounding_boxes') or []
        for i, t in enumerate(ts):
            if not (start_sec <= t <= end_sec):
                continue
            bb = bbs[i] if i < len(bbs) else None
            h = float(bb[3]) - float(bb[1]) if (bb and len(bb) == 4) else 0.0
            out.append((float(t), max(h, 0.0), bb))
    out.sort(key=lambda x: x[0])
    return out


def _kernel_scores(dets: List[Tuple[float, float, list]]) -> List[Tuple[float, float]]:
    """`(score, t)` per detection timestamp: Gaussian-kernel-weighted sum of
    bbox height over neighbors within +/-KERNEL_SUPPORT_SEC. Rewards a moment
    where bystanders are CLOSE (tall boxes) and PERSISTENTLY present, not a
    single-frame spike. Descending by score."""
    ts = np.array([t for t, _, _ in dets])
    hs = np.array([h for _, h, _ in dets])
    scored = []
    for t in ts:
        m = np.abs(ts - t) <= KERNEL_SUPPORT_SEC
        s = float(np.sum(hs[m] * np.exp(-((ts[m] - t) / KERNEL_SIGMA_SEC) ** 2)))
        scored.append((s, float(t)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


def _straddle_window(climax_sec: float, duration_sec: float) -> list:
    """`[climax-1, climax+3]`, SHIFTED (not clipped) to stay inside
    [0, duration] — a window near the clip edge keeps its full span."""
    span = WINDOW_PRE_SEC + WINDOW_POST_SEC
    lo = climax_sec - WINDOW_PRE_SEC
    hi = climax_sec + WINDOW_POST_SEC
    if lo < 0:
        lo, hi = 0.0, min(span, duration_sec)
    elif hi > duration_sec:
        hi = duration_sec
        lo = max(0.0, duration_sec - span)
    return [round(lo, 2), round(hi, 2)]


class _FaceVerifier:
    """Lazy per-process BlazeFace wrapper. `best_face_px(frame, bbox)` returns
    the tallest detected face (px, conf) in the padded bystander crop."""

    def __init__(self):
        self._detector = None
        self.available = mp is not None and FACE_MODEL.exists()

    def _get(self):
        if self._detector is None:
            self._detector = mp_vision.FaceDetector.create_from_options(
                mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_detection_confidence=FACE_DETECT_CONF,
                ))
        return self._detector

    def best_face_px(self, frame, bbox) -> Tuple[int, float]:
        import cv2
        h, w = frame.shape[:2]
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frame[max(0, y1 - CROP_PAD_PX):min(h, y2 + CROP_PAD_PX),
                         max(0, x1 - CROP_PAD_PX):min(w, x2 + CROP_PAD_PX)]
        else:
            crop = frame
        if crop.size == 0:
            return 0, 0.0
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        det = self._get().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        best_px, best_conf = 0, 0.0
        for d in det.detections:
            if d.bounding_box.height > best_px:
                best_px = int(d.bounding_box.height)
                best_conf = round(float(d.categories[0].score), 3)
        return best_px, best_conf


def _verify_candidates(cap, fps: float, candidates: List[Tuple[float, float]],
                       dets: List[Tuple[float, float, list]],
                       verifier: _FaceVerifier) -> Optional[Tuple[float, int, float]]:
    """Decode ONE frame per kernel candidate and BlazeFace the tallest nearby
    bystander crop. Returns `(climax_sec, face_px, face_conf)` for the candidate
    with the largest resolvable face, or None if no candidate shows a face (the
    caller then falls back to the top kernel score). Random seek is correct
    here — a handful of widely-spaced frames, exactly like the shared
    face-quality prefilter (its measured cost model: ~40 ms/frame)."""
    import cv2
    best: Optional[Tuple[float, int, float]] = None
    for _, cand_t in candidates:
        cap.set(cv2.CAP_PROP_POS_MSEC, cand_t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        # tallest bystander box within the kernel support of this candidate
        near = [(h, bb) for t, h, bb in dets if abs(t - cand_t) <= KERNEL_SUPPORT_SEC]
        bbox = max(near, key=lambda x: x[0])[1] if near else None
        px, conf = verifier.best_face_px(frame, bbox)
        if px > 0 and (best is None or px > best[1]):
            best = (cand_t, px, conf)
    return best


def _caption_segment(cap, climax_sec: float, task_label: str,
                     vlm_model: str) -> Optional[str]:
    """One-line wearer-action caption for a segment (docs/06 Issue 1).

    Decodes CAPTION_FRAME_OFFSETS_SEC frames around the climax and asks the
    caption VLM what the WEARER is doing. The prompt provides two sentinel
    escapes so the model can decline instead of hallucinating: 'conversation'
    (talking/listening only — the common Ego4D case flagged in the Issue 5
    brainstorm) and 'unclear'. Best-effort: any failure logs and returns None
    (field simply absent; every other segment output is unaffected)."""
    import cv2
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            img_paths = []
            for i, off in enumerate(CAPTION_FRAME_OFFSETS_SEC):
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, climax_sec + off) * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                if max(h, w) > CAPTION_MAX_EDGE_PX:
                    scale = CAPTION_MAX_EDGE_PX / max(h, w)
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                p = str(Path(temp_dir) / f"cap_{i}.jpg")
                cv2.imwrite(p, frame)
                img_paths.append(p)
            if not img_paths:
                return None
            prompt = (
                "These frames are from a head-mounted camera (you see what the "
                f"wearer sees). The overall activity is: '{task_label}'. "
                "In one phrase of 5 to 12 words, describe the physical action the "
                "camera wearer is performing RIGHT NOW, including who or what it is "
                "directed at (e.g. 'hands a card to the player opposite'). "
                "If the wearer is only talking or listening with no "
                "discrete physical action, reply exactly 'conversation'. If you "
                "cannot tell, reply exactly 'unclear'. Reply with ONLY the action "
                "phrase — no explanation, no preamble."
            )
            # temperature 0 = deterministic labels; num_predict bounds a
            # rambling decode (the smoke test saw 25 s spent explaining its way
            # to 'unclear' — the cap turns that into a fast, parseable answer).
            out = ollama_chat(vlm_model, prompt, image_paths=img_paths,
                              options={"temperature": 0, "num_predict": 48},
                              timeout=CAPTION_VLM_TIMEOUT)
            caption = " ".join(out.strip().split())
            if not caption:
                return None
            low = caption.lower().strip(" .!'\"")
            if low in CAPTION_SENTINELS:
                return low
            # Sentinel salvage: a verbose model that ENDS on a sentinel
            # ("...the answer is: unclear.") still means the sentinel.
            last = low.split()[-1].strip(" .!:'\"") if low.split() else ""
            if last in CAPTION_SENTINELS:
                return last
            return caption
    except Exception as e:
        print(f"[02b] caption failed @{climax_sec:.1f}s: {e}")
        return None


def _present_unreactive_controls(clusters, chosen_climaxes, dets) -> List[Tuple[float, list]]:
    """Type-(a) controls: for each cluster, the detection timestamp FARTHEST
    from that cluster's chosen climax (>= CONTROL_MIN_CLIMAX_SEPARATION_SEC away).
    Returns up to CONTROL_MAX_PRESENT `(anchor_t, cluster)` picks, farthest first."""
    picks = []
    for (cs, ce, n), climax in zip(clusters, chosen_climaxes):
        cands = [t for t, _, _ in dets if cs <= t <= ce]
        if not cands:
            continue
        far_t = max(cands, key=lambda t: abs(t - climax))
        if abs(far_t - climax) >= CONTROL_MIN_CLIMAX_SEPARATION_SEC:
            picks.append((far_t, [cs, ce, n]))
    picks.sort(key=lambda p: -abs(p[0] - p[1][0]))  # widest separation first
    return picks[:CONTROL_MAX_PRESENT]


def _no_audience_controls(clusters, dets, start_sec, end_sec) -> List[float]:
    """Type-(b) controls: the midpoint of the largest bystander-free gap
    (>= CONTROL_MIN_GAP_SEC) between clusters / task edges, verified to have no
    detection within CONTROL_CLEARANCE_SEC of the window span."""
    bounds = [start_sec] + [b for cs, ce, _ in clusters for b in (cs, ce)] + [end_sec]
    gaps = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds) - 1, 2)]
    gaps = [(a, b) for a, b in gaps if (b - a) >= CONTROL_MIN_GAP_SEC]
    gaps.sort(key=lambda g: -(g[1] - g[0]))
    det_ts = [t for t, _, _ in dets]
    out = []
    for a, b in gaps:
        mid = (a + b) / 2.0
        w_lo = mid - WINDOW_PRE_SEC - CONTROL_CLEARANCE_SEC
        w_hi = mid + WINDOW_POST_SEC + CONTROL_CLEARANCE_SEC
        if not any(w_lo <= t <= w_hi for t in det_ts):
            out.append(mid)
        if len(out) >= CONTROL_MAX_NO_AUDIENCE:
            break
    return out


def compute_task_climax_for_video(
    cap,
    fps: float,
    total_frames: int,
    tasks: Iterable[dict],
    duration_sec: float,
    bystander_detections: Optional[list] = None,
    vlm_model: Optional[str] = None,   # caption model (docs/06 Issue 1); None = no captions
    skip_vlm: bool = False,            # True disables ALL VLM work (captions)
    face_verify: Optional[bool] = None,
    control_segments: Optional[bool] = None,
) -> None:
    """Populate `task_temporal_metadata` on each task in-place. Tasks that
    already have non-empty metadata are skipped, so this stays safe to invoke
    from multiple Layer 03 pipelines: only the first call does the work.

    `cap` (an open cv2.VideoCapture) is used only for face verification and
    action captions; pass `face_verify=False` and no `vlm_model` for a
    pure-manifest pass that never touches the video. `vlm_model`/`skip_vlm`
    keep their flow-era call-site signatures, repurposed for the caption pass
    (docs/06 Issue 1): every pre-existing caller passes `skip_vlm=True`, which
    correctly disables captions on the lazy/back-compat path."""
    if face_verify is None:
        face_verify = FACE_VERIFY
    if control_segments is None:
        control_segments = CONTROL_SEGMENTS
    verifier = _FaceVerifier() if face_verify else None
    do_verify = bool(verifier and verifier.available and cap is not None)
    do_caption = bool(vlm_model and not skip_vlm and ollama_chat is not None
                      and cap is not None)

    for task in tasks:
        if task.get('task_temporal_metadata'):
            continue

        start_sec = task['task_start_sec']
        end_sec = task['task_end_sec']

        dets = _detections_in(bystander_detections, start_sec, end_sec)
        clusters = _cluster_timestamps([t for t, _, _ in dets],
                                       CLIMAX_CLUSTER_GAP_SEC, CLIMAX_MAX_CLUSTER_SPAN_SEC)
        clusters = [c for c in clusters if c[2] >= CLIMAX_MIN_CLUSTER_DETECTIONS]
        clusters.sort(key=lambda c: c[2], reverse=True)   # densest first
        clusters = clusters[:CLIMAX_MAX_SEGMENTS]
        clusters.sort(key=lambda c: c[0])                 # back to chronological order

        def _finish_segment(seg, climax_sec):
            if do_caption:
                caption = _caption_segment(cap, climax_sec,
                                           task.get('task_label', 'unknown'), vlm_model)
                if caption is not None:
                    seg["segment_action_caption"] = caption
            return seg

        segments: List[dict] = []
        produced: List[Tuple[list, float]] = []   # (cluster, chosen_climax) for controls
        for cs, ce, n in clusters:
            cluster_dets = [(t, h, bb) for t, h, bb in dets if cs <= t <= ce]
            if not cluster_dets:
                continue
            scored = _kernel_scores(cluster_dets)
            climax_sec, kernel_score = scored[0][1], scored[0][0]
            method = "bbox_kernel_peak"
            face_px, face_conf = 0, 0.0

            if do_verify and len(scored) > 1:
                chosen = _verify_candidates(cap, fps, scored[:FACE_VERIFY_CANDIDATES],
                                            cluster_dets, verifier)
                if chosen is not None:
                    climax_sec, face_px, face_conf = chosen
                    kernel_score = next(s for s, t in scored if t == climax_sec)
                    method = "bbox_kernel_peak+face_verified"

            seg = {
                "task_climax_sec": round(climax_sec, 2),
                "task_reaction_window_sec": _straddle_window(climax_sec, duration_sec),
                "climax_extraction_method": method,
                "bbox_kernel_score": round(float(kernel_score), 1),
                "segment_face_px": int(face_px),
                "bystander_cluster_sec": [round(cs, 2), round(ce, 2)],
                "cluster_detection_count": int(n),
            }
            if face_px:
                seg["segment_face_conf"] = face_conf
            segments.append(_finish_segment(seg, climax_sec))
            produced.append(([cs, ce, n], climax_sec))

        n_real = len(segments)

        # Control (negative) segments — docs/06 Issue 3. APPENDED after the
        # real segments so segment_index alignment is stable everywhere.
        if control_segments:
            used_clusters = [c for c, _ in produced]
            chosen_climaxes = [cx for _, cx in produced]
            for anchor_t, cl in _present_unreactive_controls(
                    used_clusters, chosen_climaxes, dets):
                face_px, face_conf = 0, 0.0
                if do_verify:
                    near = [(t, h, bb) for t, h, bb in dets
                            if abs(t - anchor_t) <= KERNEL_SUPPORT_SEC]
                    chosen = _verify_candidates(cap, fps, [(0.0, anchor_t)],
                                                near or dets, verifier)
                    if chosen is not None:
                        _, face_px, face_conf = chosen
                seg = {
                    "task_climax_sec": round(anchor_t, 2),
                    "task_reaction_window_sec": _straddle_window(anchor_t, duration_sec),
                    "climax_extraction_method": "control_present_unreactive",
                    "is_control": True,
                    "control_type": "present_unreactive",
                    "segment_face_px": int(face_px),
                    "bystander_cluster_sec": [round(cl[0], 2), round(cl[1], 2)],
                    "cluster_detection_count": int(cl[2]),
                }
                if face_px:
                    seg["segment_face_conf"] = face_conf
                segments.append(_finish_segment(seg, anchor_t))
            for anchor_t in _no_audience_controls(used_clusters, dets, start_sec, end_sec):
                seg = {
                    "task_climax_sec": round(anchor_t, 2),
                    "task_reaction_window_sec": _straddle_window(anchor_t, duration_sec),
                    "climax_extraction_method": "control_no_audience",
                    "is_control": True,
                    "control_type": "no_audience",
                    "segment_face_px": 0,
                    "cluster_detection_count": 0,
                }
                segments.append(_finish_segment(seg, anchor_t))

        meta = {
            "reaction_segments": segments,
            "n_reaction_segments": len(segments),
            "n_control_segments": len(segments) - n_real,
        }
        real = segments[:n_real]
        if real:
            # Mirror the densest REAL segment at the top level so any legacy
            # single-window reader still gets a sensible (bystander-aligned,
            # non-control) window.
            primary = max(real, key=lambda s: s["cluster_detection_count"])
            meta["task_climax_sec"] = primary["task_climax_sec"]
            meta["task_reaction_window_sec"] = primary["task_reaction_window_sec"]
            meta["climax_extraction_method"] = primary["climax_extraction_method"]
        else:
            meta["climax_extraction_method"] = "no_bystander_cluster_in_task"
        task['task_temporal_metadata'] = meta


def _entry_needs_climax(entry) -> bool:
    tasks = entry.get('identified_tasks', [])
    return bool(tasks) and not all(t.get('task_temporal_metadata') for t in tasks)


def _annotate_one_entry(args):
    """Worker for the parallel path. Module-level so it is picklable under the
    macOS 'spawn' start method. `opts` is `{"face_verify": bool,
    "caption_model": Optional[str]}`. Returns (entry, updated_bool)."""
    entry, opts = args
    face_verify = bool(opts.get("face_verify"))
    caption_model = opts.get("caption_model")
    control_segments = opts.get("control_segments")
    if not _entry_needs_climax(entry):
        return entry, False

    import cv2
    try:
        cv2.setNumThreads(1)  # each worker single-threaded; parallelism is across clips
    except Exception:
        pass

    cap = None
    fps = entry.get('fps') or 0.0
    duration_sec = entry.get('duration_sec') or 0.0
    video_path = entry.get('video_path')
    needs_decode = face_verify or bool(caption_model)
    if needs_decode and video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            if not duration_sec and fps:
                duration_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        else:
            cap.release()
            cap = None
    if not duration_sec:
        # Manifest-only fallback: bound windows by the last known detection
        # rather than dropping the entry (the shift-at-edge only needs an upper
        # bound; overshooting is harmless for in-range climaxes).
        last_det = max((t for tr in (entry.get('bystander_detections') or [])
                        for t in (tr.get('timestamps_sec') or [])), default=0.0)
        duration_sec = max((t.get('task_end_sec', 0.0) for t in entry.get('identified_tasks', [])),
                           default=last_det) or last_det
    if not duration_sec:
        return entry, False

    try:
        compute_task_climax_for_video(
            cap, fps, 0, entry['identified_tasks'], duration_sec,
            bystander_detections=entry.get('bystander_detections'),
            face_verify=face_verify and cap is not None,
            vlm_model=caption_model if cap is not None else None,
            control_segments=control_segments,
        )
        return entry, True
    finally:
        if cap is not None:
            cap.release()


class TaskClimaxPipeline:
    """Layer 02b pipeline: annotate a Node-02 manifest with reaction segments.

    Follows the Layer 03 Failure & Resumability Policy (docs/03): idempotent
    resume (only entries with un-annotated tasks are processed), per-entry
    failure isolation, and atomic checkpointed writes back to the manifest.
    """

    def __init__(self, manifest_path, force: bool = False,
                 workers: Optional[int] = None, face_verify: Optional[bool] = None,
                 entry_filter: Optional[Callable[[dict], bool]] = None,
                 action_captions: Optional[bool] = None,
                 control_segments: Optional[bool] = None):
        self.manifest_path = Path(manifest_path)
        self.force = force
        self.workers = max(1, int(workers)) if workers else 1
        self.face_verify = FACE_VERIFY if face_verify is None else face_verify
        self.control_segments = CONTROL_SEGMENTS if control_segments is None else control_segments
        self.entry_filter = entry_filter
        self.error_log_path = self.manifest_path.parent / "02b_task_climax_errors.json"
        # Action captions (docs/06 Issue 1): resolve the caption VLM from the
        # central tier registry. Degrades to no-captions (logged) if the
        # registry/vlm_client are unavailable — never blocks the annotation.
        if action_captions is None:
            action_captions = ACTION_CAPTIONS
        self.caption_model = None
        if action_captions:
            if get_model is None or ollama_chat is None:
                print("[02b] action captions requested but models_config/vlm_client "
                      "unavailable — captions skipped.")
            else:
                try:
                    self.caption_model = get_model("filtering_vlm")
                except Exception as e:
                    print(f"[02b] caption model resolution failed ({e}) — captions skipped.")

    def _log_error(self, video_id, error):
        errors = []
        if self.error_log_path.exists():
            try:
                with open(self.error_log_path) as f:
                    errors = json.load(f)
            except Exception:
                pass
        errors.append({"video_id": video_id, "error": str(error)})
        tmp = self.error_log_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(errors, f, indent=4)
        tmp.replace(self.error_log_path)

    def run(self) -> int:
        """Annotate every entry that needs it. Returns entries updated."""
        if not self.manifest_path.exists():
            print(f"[02b] manifest not found: {self.manifest_path}")
            return 0
        with open(self.manifest_path) as f:
            entries = json.load(f)

        if self.force:
            for e in entries:
                for t in e.get('identified_tasks', []):
                    t['task_temporal_metadata'] = {}

        todo = [i for i, e in enumerate(entries)
                if _entry_needs_climax(e)
                and (self.entry_filter is None or self.entry_filter(e))]
        if not todo:
            print("[02b] nothing to do (all tasks annotated).")
            return 0

        args = [(entries[i], {"face_verify": self.face_verify,
                              "caption_model": self.caption_model,
                              "control_segments": self.control_segments}) for i in todo]
        updated = 0

        def _checkpoint():
            # Atomic write (temp + rename) so a crash mid-write cannot corrupt
            # the manifest; combined with the idempotent skip this makes the
            # pass crash-resumable.
            tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
            with open(tmp, 'w') as f:
                json.dump(entries, f, indent=4)
            os.replace(tmp, self.manifest_path)

        if self.workers > 1 and len(args) > 1:
            # imap (ordered, lazy) so we can checkpoint mid-stream.
            with Pool(processes=min(self.workers, len(args))) as pool:
                for done, (idx, (entry, was_updated)) in enumerate(
                        zip(todo, pool.imap(_annotate_one_entry, args)), 1):
                    entries[idx] = entry
                    updated += 1 if was_updated else 0
                    if done % CLIMAX_CHECKPOINT_EVERY == 0:
                        _checkpoint()
        else:
            for done, (idx, a) in enumerate(zip(todo, args), 1):
                try:
                    entry, was_updated = _annotate_one_entry(a)
                except Exception as e:  # per-entry isolation (policy #1)
                    self._log_error(entries[idx].get('id', entries[idx].get('video_id', '?')), e)
                    continue
                entries[idx] = entry
                updated += 1 if was_updated else 0
                if done % CLIMAX_CHECKPOINT_EVERY == 0:
                    _checkpoint()

        if updated:
            _checkpoint()
        return updated


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Layer 02b: annotate a Node-02 manifest with bystander-anchored "
                    "reaction segments (bbox-kernel climax + face verification).")
    ap.add_argument("manifest")
    ap.add_argument("--workers", type=int,
                    default=int(os.getenv("SR_CLIMAX_WORKERS", "0")) or max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--no-face-verify", action="store_true",
                    help="Skip the per-segment BlazeFace verification.")
    ap.add_argument("--no-captions", action="store_true",
                    help="Skip the per-segment VLM action captions (docs/06 Issue 1). "
                         "Captions need a running ollama server and dominate 02b "
                         "runtime (~12s/segment cold, serialized by the single GPU).")
    ap.add_argument("--no-controls", action="store_true",
                    help="Skip the control (negative) segments (docs/06 Issue 3).")
    ap.add_argument("--force", action="store_true",
                    help="Re-annotate entries that already have task_temporal_metadata.")
    a = ap.parse_args()
    t0 = time.time()
    pipe = TaskClimaxPipeline(a.manifest, force=a.force, workers=a.workers,
                              face_verify=not a.no_face_verify,
                              action_captions=not a.no_captions,
                              control_segments=not a.no_controls)
    n = pipe.run()
    print(f"[02b] annotated {n} entries with {a.workers} workers in {time.time()-t0:.0f}s -> {a.manifest}")
