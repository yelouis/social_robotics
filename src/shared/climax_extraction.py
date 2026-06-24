"""Shared task-climax extraction utility.

Per Resolved Issue #8 (May 17), `task_temporal_metadata` is no longer
populated by Layer 02. Layer 02 now emits `identified_tasks` with an empty
`task_temporal_metadata = {}`, and any Layer 03 pipeline that consumes the
reaction window calls `populate_climax_for_manifest()` (or, if it already
holds an open `cv2.VideoCapture`, `compute_task_climax_for_video()`) before
its own feature extraction. The first Layer 03 to run for a given manifest
fills in the metadata in place; subsequent layers find it cached and skip
the optical-flow pass.

**Bystander-aware multi-window rework (June 23).** The original detector
maximized optical flow over the ENTIRE `[task_start, task_end]` range and
emitted a single short reaction window. On long egocentric clips (median ~26-min
task ranges on the production corpus) that picks one *wearer*-motion peak
anywhere in the task — frequently nowhere near a bystander — so downstream layers
sampling that window find nothing (03e scored 0/200 on the resulting windows).
Instead we now cluster the task's **bystander detections** in time and emit one
reaction *segment* per cluster: a long clip yields multiple social-moment windows
and the optical-flow search is bounded to where a bystander is actually present.
Each task's `task_temporal_metadata` carries a `reaction_segments` list, with the
densest segment mirrored at the top level for backward-compatible single-window
readers. Consumers should iterate `iter_reaction_windows(task)`.
"""

import json
import os
import re
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Callable

import cv2
import numpy as np

try:
    from .vlm_client import ollama_chat
except ImportError:
    from shared.vlm_client import ollama_chat

# Generous safety bound for a TRUE VLM deadlock during climax refinement (a
# normal call is a few seconds); used as the enforced httpx timeout in
# ollama_chat. On timeout the connection closes and the generation is cancelled.
CLIMAX_VLM_REQUEST_TIMEOUT = 180

# --- Bystander-aware multi-window climax (rework) ---
CLIMAX_CLUSTER_GAP_SEC = 15.0        # split clusters where bystanders are absent > this
CLIMAX_MAX_CLUSTER_SPAN_SEC = 90.0   # also split clusters longer than this (bounds per-segment flow)
CLIMAX_MIN_CLUSTER_DETECTIONS = 2    # ignore singleton/spurious clusters (1-detection fragments)
CLIMAX_MAX_SEGMENTS = 10             # cap segments/task (densest kept) — bounds cost + output size
CLIMAX_CLUSTER_PAD_SEC = 2.0         # pad each cluster span for the optical-flow search
CLIMAX_CHECKPOINT_EVERY = 15         # write the manifest every N entries (crash-resumable pre-pass)


def _reaction_window(climax_sec: float, velocity: str, duration_sec: float) -> list:
    if velocity == 'fast':
        window = [round(climax_sec + 0.5, 2), round(climax_sec + 2.0, 2)]
    elif velocity == 'medium':
        window = [round(climax_sec + 1.0, 2), round(climax_sec + 3.0, 2)]
    else:
        window = [round(climax_sec + 2.0, 2), round(climax_sec + 6.0, 2)]
    duration_rounded = round(duration_sec, 2)
    return [min(window[0], duration_rounded), min(window[1], duration_rounded)]


def _bystander_timestamps_in(bystander_detections, start_sec: float, end_sec: float) -> List[float]:
    """Flatten every bystander track's detection timestamps that fall inside the
    task's [start, end] range. `bystander_detections` is Node-02's per-person
    track list; each track carries co-indexed `timestamps_sec`."""
    out: List[float] = []
    for track in (bystander_detections or []):
        for t in (track.get('timestamps_sec') or []):
            if start_sec <= t <= end_sec:
                out.append(float(t))
    return out


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


def _optical_flow_climax(cap: cv2.VideoCapture, fps: float,
                         range_start_frame: int, range_end_frame: int):
    """Coarse (~5 FPS grab-skip) + dense (±1 s) Farneback optical-flow peak within
    a bounded frame range. Returns `(climax_frame, max_flow, flow_data)` or
    `(None, 0.0, [])` if the range cannot be read.

    Sequential decode (climax-speedup Resolved Issue): the coarse loop `grab()`s
    forward cheaply and `read()`s only every-`step`-th frame it analyzes (the old
    per-iteration `cap.set(POS_FRAMES)` re-decodes from the nearest keyframe on
    H.264, ~10-20x wasted decode). The exact same frames feed Farneback, so the
    peak is unchanged."""
    step = max(1, int(fps / 5))
    cap.set(cv2.CAP_PROP_POS_FRAMES, range_start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        return None, 0.0, []
    prev_gray = cv2.resize(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)

    max_flow = 0.0
    climax_frame = range_start_frame
    flow_data: List[Tuple[int, float]] = []
    pos = range_start_frame + 1  # next frame index the capture will return
    for frame_idx in range(range_start_frame + step, range_end_frame, step):
        ok = True
        while pos < frame_idx:
            if not cap.grab():
                ok = False
                break
            pos += 1
        if not ok or pos != frame_idx:
            break
        ret, frame = cap.read()
        if not ret:
            break
        pos = frame_idx + 1
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        flow_data.append((frame_idx, mean_mag))
        if mean_mag > max_flow:
            max_flow = mean_mag
            climax_frame = frame_idx
        prev_gray = gray

    # Dense refine: pure sequential read across the ±1 s window around the peak.
    window_frames = int(1.0 * fps)
    dense_start = max(range_start_frame, climax_frame - window_frames)
    dense_end = min(range_end_frame, climax_frame + window_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, dense_start)
    ret, prev_dense_frame = cap.read()
    if ret:
        prev_dense_gray = cv2.resize(cv2.cvtColor(prev_dense_frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)
        dense_max_flow = 0.0
        dense_climax_frame = dense_start
        for frame_idx in range(dense_start + 1, dense_end):
            ret, dense_frame = cap.read()
            if not ret:
                break
            dense_gray = cv2.resize(cv2.cvtColor(dense_frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5)
            flow = cv2.calcOpticalFlowFarneback(prev_dense_gray, dense_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            if mean_mag > dense_max_flow:
                dense_max_flow = mean_mag
                dense_climax_frame = frame_idx
            prev_dense_gray = dense_gray
        climax_frame = dense_climax_frame
        if dense_max_flow > 0:
            max_flow = dense_max_flow
    return climax_frame, max_flow, flow_data


def _vlm_refine_climax(cap, fps, candidates, task, vlm_model) -> Optional[int]:
    """Slow-task VLM refinement: ask the VLM which of the top-flow candidate
    frames best represents the action climax. Returns the chosen frame index or
    None. Best-effort — any failure logs and returns None."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            img_paths = []
            for i, (cand_frame, _) in enumerate(candidates):
                cap.set(cv2.CAP_PROP_POS_FRAMES, cand_frame)
                ret, frame = cap.read()
                if not ret:
                    continue
                h, w = frame.shape[:2]
                if max(h, w) > 1024:
                    scale = 1024 / max(h, w)
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                img_path = temp_path / f"cand_{i + 1}.jpg"
                cv2.imwrite(str(img_path), frame)
                img_paths.append(str(img_path))
            if not img_paths:
                return None
            prompt = (
                f"The person is performing the task: '{task['task_label']}'. "
                f"I have provided {len(img_paths)} images from the video. "
                "Which image (respond with just the number 1, 2, or 3) "
                "best represents the 'climax' or the most critical moment of this action? "
                "If you are unsure, pick the one with the most active motion."
            )
            content = ollama_chat(vlm_model, prompt, image_paths=img_paths,
                                  timeout=CLIMAX_VLM_REQUEST_TIMEOUT).strip()
            matches = re.findall(r'[1-3]', content)
            if matches:
                choice = int(matches[0]) - 1
                if choice < len(candidates):
                    return candidates[choice][0]
    except Exception as e:
        print(f"[climax_extraction] VLM refinement failed: {e}")
    return None


def compute_task_climax_for_video(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    tasks: Iterable[dict],
    duration_sec: float,
    bystander_detections: Optional[list] = None,
    vlm_model: Optional[str] = None,
    skip_vlm: bool = False,
) -> None:
    """Populate `task_temporal_metadata` on each task in-place (bystander-aware
    multi-window). Tasks that already have non-empty metadata are skipped, so
    this is safe to invoke from multiple Layer 03 pipelines: only the first call
    does the work.

    For each task we cluster its bystander detections in time and emit one
    reaction *segment* per cluster (densest `CLIMAX_MAX_SEGMENTS` kept), computing
    the optical-flow climax inside each cluster's (padded, span-capped) range.
    The result is written as `{"reaction_segments": [...], ...}` with the densest
    segment's fields mirrored at the top level for legacy single-window readers.
    A task whose range contains no qualifying bystander cluster gets an empty
    `reaction_segments` (no social signal to window)."""
    for task in tasks:
        if task.get('task_temporal_metadata'):
            continue

        start_sec = task['task_start_sec']
        end_sec = task['task_end_sec']
        velocity = task.get('task_velocity', 'medium')

        # Cluster the task's bystander detections in time → social-moment segments.
        ts = _bystander_timestamps_in(bystander_detections, start_sec, end_sec)
        clusters = _cluster_timestamps(ts, CLIMAX_CLUSTER_GAP_SEC, CLIMAX_MAX_CLUSTER_SPAN_SEC)
        clusters = [c for c in clusters if c[2] >= CLIMAX_MIN_CLUSTER_DETECTIONS]
        clusters.sort(key=lambda c: c[2], reverse=True)   # densest first
        clusters = clusters[:CLIMAX_MAX_SEGMENTS]
        clusters.sort(key=lambda c: c[0])                 # back to chronological order

        segments: List[dict] = []
        for cs, ce, n in clusters:
            r_start = max(int(start_sec * fps), int((cs - CLIMAX_CLUSTER_PAD_SEC) * fps))
            r_end = min(int(end_sec * fps), int((ce + CLIMAX_CLUSTER_PAD_SEC) * fps) + 1)
            if r_end <= r_start + 1:
                # degenerate (single-instant) cluster — center the climax on it
                climax_frame = int(cs * fps)
                max_flow = 0.0
                flow_data: List[Tuple[int, float]] = []
            else:
                climax_frame, max_flow, flow_data = _optical_flow_climax(cap, fps, r_start, r_end)
                if climax_frame is None:
                    climax_frame = int(((cs + ce) / 2.0) * fps)
                    max_flow = 0.0
                    flow_data = []

            climax_sec = climax_frame / fps
            method = "optical_flow_peak_only" if skip_vlm else "optical_flow_peak"
            vlm_confidence: Optional[float] = None
            if not skip_vlm and velocity == 'slow' and len(flow_data) > 1 and vlm_model:
                candidates = sorted(flow_data, key=lambda x: x[1], reverse=True)[:3]
                candidates = sorted(candidates, key=lambda x: x[0])
                chosen = _vlm_refine_climax(cap, fps, candidates, task, vlm_model)
                if chosen is not None:
                    climax_frame = chosen
                    climax_sec = climax_frame / fps
                    method = "optical_flow_peak+vlm_refinement"
                    vlm_confidence = 1.0

            seg = {
                "task_climax_sec": round(climax_sec, 2),
                "task_reaction_window_sec": _reaction_window(climax_sec, velocity, duration_sec),
                "climax_extraction_method": method,
                "optical_flow_peak_magnitude": round(float(max_flow), 2),
                "bystander_cluster_sec": [round(cs, 2), round(ce, 2)],
                "cluster_detection_count": int(n),
            }
            if vlm_confidence is not None:
                seg["vlm_climax_confidence"] = vlm_confidence
            segments.append(seg)

        meta = {
            "reaction_segments": segments,
            "n_reaction_segments": len(segments),
        }
        if segments:
            # Mirror the densest segment at the top level so any legacy
            # single-window reader still gets a sensible (bystander-aligned) window.
            primary = max(segments, key=lambda s: s["cluster_detection_count"])
            meta["task_climax_sec"] = primary["task_climax_sec"]
            meta["task_reaction_window_sec"] = primary["task_reaction_window_sec"]
            meta["climax_extraction_method"] = primary["climax_extraction_method"]
            meta["optical_flow_peak_magnitude"] = primary["optical_flow_peak_magnitude"]
            if "vlm_climax_confidence" in primary:
                meta["vlm_climax_confidence"] = primary["vlm_climax_confidence"]
        else:
            meta["climax_extraction_method"] = "no_bystander_cluster_in_task"
        task['task_temporal_metadata'] = meta


def iter_reaction_windows(task: dict) -> Iterator[Tuple[float, list]]:
    """Yield `(climax_sec, [start, end])` for each reaction segment of a task.

    This is the canonical way for Layer 03 pipelines to consume reaction windows:
    it transparently handles the multi-segment schema (one window per bystander
    cluster) and the legacy single-window format. A task with no usable window
    yields nothing."""
    meta = task.get('task_temporal_metadata') or {}
    segments = meta.get('reaction_segments')
    if segments:
        for s in segments:
            w = s.get('task_reaction_window_sec')
            if w and len(w) == 2:
                yield s.get('task_climax_sec', (w[0] + w[1]) / 2.0), w
    else:
        w = meta.get('task_reaction_window_sec')
        if w and len(w) == 2:
            yield meta.get('task_climax_sec', (w[0] + w[1]) / 2.0), w


def expand_task_segments(tasks: Iterable[dict]) -> Iterator[dict]:
    """Yield one shallow-copied 'pseudo-task' per reaction segment, each carrying
    a *single-window* `task_temporal_metadata`, so an existing per-task loop
    becomes per-segment with no restructuring — replace `for task in tasks:` with
    `for task in expand_task_segments(tasks):` and the body is unchanged. Each
    yielded task keeps all original fields (`task_id`, `task_start_sec`,
    `task_velocity`, …) and adds `segment_index` / `n_segments`. A task whose
    range produced no usable reaction window yields nothing (correctly skipped)."""
    for task in tasks:
        windows = list(iter_reaction_windows(task))
        n = len(windows)
        for i, (climax_sec, window) in enumerate(windows):
            pseudo = dict(task)
            base_meta = dict(task.get('task_temporal_metadata') or {})
            base_meta.pop('reaction_segments', None)  # collapse to a single-window view
            base_meta['task_reaction_window_sec'] = window
            base_meta['task_climax_sec'] = climax_sec
            pseudo['task_temporal_metadata'] = base_meta
            pseudo['segment_index'] = i
            pseudo['n_segments'] = n
            yield pseudo


def _entry_needs_climax(entry) -> bool:
    tasks = entry.get('identified_tasks', [])
    return bool(tasks) and not all(t.get('task_temporal_metadata') for t in tasks)


def _populate_one_entry(args):
    """Worker for the parallel path. Module-level so it is picklable under the
    macOS 'spawn' start method. Computes climax for one entry's tasks in place
    and returns (entry, updated_bool)."""
    entry, vlm_model, skip_vlm = args
    try:
        cv2.setNumThreads(1)  # each worker single-threaded; parallelism is across clips
    except Exception:
        pass
    if not _entry_needs_climax(entry):
        return entry, False
    video_path = entry.get('video_path')
    if not video_path or not Path(video_path).exists():
        return entry, False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return entry, False
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or entry.get('fps') or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = entry.get('duration_sec') or (total_frames / fps if fps else 0.0)
        if fps <= 0 or duration_sec <= 0:
            return entry, False
        compute_task_climax_for_video(
            cap, fps, total_frames, entry['identified_tasks'], duration_sec,
            bystander_detections=entry.get('bystander_detections'),
            vlm_model=vlm_model, skip_vlm=skip_vlm,
        )
        return entry, True
    finally:
        cap.release()


def populate_climax_for_manifest(
    manifest_path: Path,
    vlm_model: Optional[str] = None,
    skip_vlm: bool = False,
    workers: Optional[int] = None,
    entry_filter: Optional[Callable[[dict], bool]] = None,
) -> int:
    """Fill in `task_temporal_metadata` for every entry in `manifest_path`
    that has tasks with empty metadata. Writes back to the same path. Returns
    the number of entries updated. Idempotent — a no-op once every task has
    metadata.

    Climax is per-clip independent, so entries can be processed in parallel.
    `workers` controls process count: **default 1 (serial)** so library callers
    are unaffected and spawn-safe. Pass `workers>1` (or use the guarded
    `python -m shared.climax_extraction` CLI) to parallelize the full-corpus
    pre-pass. The VLM-refinement path is only parallel-safe if the local LLM
    server tolerates concurrent calls; the optical-flow-only path (skip_vlm or
    no vlm_model) always is.

    `entry_filter` is an optional predicate `(entry) -> bool`; entries for which
    it returns False are skipped (climax not computed). This is how the
    bystander-face-quality pre-filter (03b Resolved Issue #8) avoids paying the
    dominant optical-flow cost on clips that will score nothing. The predicate is
    applied in the main process *before* dispatch, so there is no Pool-pickling
    concern and skipped clips are never opened.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return 0
    with open(manifest_path, 'r') as f:
        entries = json.load(f)

    todo = [i for i, e in enumerate(entries)
            if _entry_needs_climax(e) and (entry_filter is None or entry_filter(e))]
    if not todo:
        return 0

    if workers is None:
        workers = 1  # serial by default; opt-in to parallel via the CLI / explicit arg
    workers = max(1, int(workers))

    args = [(entries[i], vlm_model, skip_vlm) for i in todo]
    updated = 0

    def _checkpoint():
        # Atomic write (temp + rename) so a crash mid-write cannot corrupt the
        # manifest; combined with the idempotent _entry_needs_climax skip this
        # makes the pre-pass crash-resumable — a relaunch picks up where it left off.
        tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with open(tmp, 'w') as f:
            json.dump(entries, f, indent=4)
        os.replace(tmp, manifest_path)

    if workers > 1 and len(args) > 1:
        # imap (ordered, lazy) so we can checkpoint mid-stream rather than only
        # after the whole Pool finishes.
        with Pool(processes=min(workers, len(args))) as pool:
            for done, (idx, (entry, was_updated)) in enumerate(
                    zip(todo, pool.imap(_populate_one_entry, args)), 1):
                entries[idx] = entry
                updated += 1 if was_updated else 0
                if done % CLIMAX_CHECKPOINT_EVERY == 0:
                    _checkpoint()
    else:
        for done, (idx, a) in enumerate(zip(todo, args), 1):
            entry, was_updated = _populate_one_entry(a)
            entries[idx] = entry
            updated += 1 if was_updated else 0
            if done % CLIMAX_CHECKPOINT_EVERY == 0:
                _checkpoint()

    if updated:
        _checkpoint()
    return updated


if __name__ == "__main__":
    # Guarded CLI — the spawn-safe entry point for the parallel full-corpus
    # climax pre-pass. Layer runners then find climax cached and skip it.
    #   SR_CLIMAX_WORKERS=8 python -m shared.climax_extraction <manifest.json>
    import argparse
    import time as _time

    ap = argparse.ArgumentParser(description="Parallel climax pre-population for a manifest.")
    ap.add_argument("manifest")
    ap.add_argument("--workers", type=int,
                    default=int(os.getenv("SR_CLIMAX_WORKERS", "0")) or max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--no-skip-vlm", action="store_true", help="Enable slow-task VLM refinement (serial-safe only).")
    a = ap.parse_args()
    _t0 = _time.time()
    n = populate_climax_for_manifest(a.manifest, skip_vlm=not a.no_skip_vlm, workers=a.workers)
    print(f"[climax] populated {n} entries with {a.workers} workers in {_time.time()-_t0:.0f}s -> {a.manifest}")
