"""Shared task-climax extraction utility.

Per Resolved Issue #8 (May 17), `task_temporal_metadata` is no longer
populated by Layer 02. Layer 02 now emits `identified_tasks` with an empty
`task_temporal_metadata = {}`, and any Layer 03 pipeline that consumes the
reaction window calls `populate_climax_for_manifest()` (or, if it already
holds an open `cv2.VideoCapture`, `compute_task_climax_for_video()`) before
its own feature extraction. The first Layer 03 to run for a given manifest
fills in the metadata in place; subsequent layers find it cached and skip
the optical-flow pass.

The implementation mirrors the original `FilteringPipeline.temporal_climax_
identification` (two-pass coarse + dense optical flow, optional Moondream
refinement for slow tasks), but is structured as free functions so any
Layer 03 can fold it into its own sequential decode rather than re-seeking
through every Ego4D file.
"""

import json
import os
import re
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np


def _reaction_window(climax_sec: float, velocity: str, duration_sec: float) -> list:
    if velocity == 'fast':
        window = [round(climax_sec + 0.5, 2), round(climax_sec + 2.0, 2)]
    elif velocity == 'medium':
        window = [round(climax_sec + 1.0, 2), round(climax_sec + 3.0, 2)]
    else:
        window = [round(climax_sec + 2.0, 2), round(climax_sec + 6.0, 2)]
    duration_rounded = round(duration_sec, 2)
    return [min(window[0], duration_rounded), min(window[1], duration_rounded)]


def compute_task_climax_for_video(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    tasks: Iterable[dict],
    duration_sec: float,
    vlm_model: Optional[str] = None,
    skip_vlm: bool = False,
) -> None:
    """Populate `task_temporal_metadata` on each task in-place.

    Tasks that already have a non-empty `task_temporal_metadata` are skipped,
    so this is safe to invoke from multiple Layer 03 pipelines: only the
    first call does the work.
    """
    for task in tasks:
        if task.get('task_temporal_metadata'):
            continue

        start_sec = task['task_start_sec']
        end_sec = task['task_end_sec']
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        step = max(1, int(fps / 5))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = cap.read()
        if not ret:
            continue

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)

        max_flow = 0.0
        climax_frame = start_frame
        flow_data = []

        # Sequential decode (climax-speedup Resolved Issue). The old loop called
        # cap.set(POS_FRAMES, frame_idx) every iteration; on H.264 each such seek
        # re-decodes from the nearest keyframe (~tens of frames) just to land on
        # the requested one — ~10-20x wasted decode. Instead we grab() forward
        # cheaply (demux/decode without the numpy copy) and read() only the
        # every-`step`-th frame we actually analyze. The exact same frames are
        # fed to Farneback, so the optical-flow peak is unchanged.
        pos = start_frame + 1  # next frame index the capture will return
        for frame_idx in range(start_frame + step, end_frame, step):
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            flow_data.append((frame_idx, mean_mag))
            if mean_mag > max_flow:
                max_flow = mean_mag
                climax_frame = frame_idx
            prev_gray = gray

        window_frames = int(1.0 * fps)
        dense_start = max(start_frame, climax_frame - window_frames)
        dense_end = min(end_frame, climax_frame + window_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, dense_start)
        ret, prev_dense_frame = cap.read()
        if ret:
            prev_dense_gray = cv2.cvtColor(prev_dense_frame, cv2.COLOR_BGR2GRAY)
            prev_dense_gray = cv2.resize(prev_dense_gray, (0, 0), fx=0.5, fy=0.5)
            dense_max_flow = 0.0
            dense_climax_frame = dense_start
            # Dense pass walks consecutive frames, so it is a pure sequential
            # read — the cap.set(dense_start) above is the only seek needed (no
            # per-frame seeking).
            for frame_idx in range(dense_start + 1, dense_end):
                ret, dense_frame = cap.read()
                if not ret:
                    break
                dense_gray = cv2.cvtColor(dense_frame, cv2.COLOR_BGR2GRAY)
                dense_gray = cv2.resize(dense_gray, (0, 0), fx=0.5, fy=0.5)
                flow = cv2.calcOpticalFlowFarneback(prev_dense_gray, dense_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = float(np.mean(mag))
                if mean_mag > dense_max_flow:
                    dense_max_flow = mean_mag
                    dense_climax_frame = frame_idx
                prev_dense_gray = dense_gray
            climax_frame = dense_climax_frame
            max_flow = dense_max_flow if dense_max_flow > 0 else max_flow

        climax_sec = climax_frame / fps
        extraction_method = "optical_flow_peak_only" if skip_vlm else "optical_flow_peak"
        vlm_confidence: Optional[float] = None

        if not skip_vlm and task.get('task_velocity') == 'slow' and len(flow_data) > 1 and vlm_model:
            candidates = sorted(flow_data, key=lambda x: x[1], reverse=True)[:3]
            candidates = sorted(candidates, key=lambda x: x[0])
            try:
                import ollama
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    img_paths = []
                    for i, (cand_frame, _) in enumerate(candidates):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cand_frame)
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        h, w = frame.shape[:2]
                        max_dim = 1024
                        if max(h, w) > max_dim:
                            scale = max_dim / max(h, w)
                            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                        img_path = temp_path / f"cand_{i+1}.jpg"
                        cv2.imwrite(str(img_path), frame)
                        img_paths.append(str(img_path))
                    if img_paths:
                        prompt = (
                            f"The person is performing the task: '{task['task_label']}'. "
                            f"I have provided {len(img_paths)} images from the video. "
                            "Which image (respond with just the number 1, 2, or 3) "
                            "best represents the 'climax' or the most critical moment of this action? "
                            "If you are unsure, pick the one with the most active motion."
                        )
                        response = ollama.chat(
                            model=vlm_model,
                            messages=[{'role': 'user', 'content': prompt, 'images': img_paths}],
                        )
                        content = response['message']['content'].strip()
                        matches = re.findall(r'[1-3]', content)
                        if matches:
                            choice = int(matches[0]) - 1
                            if choice < len(candidates):
                                climax_frame = candidates[choice][0]
                                climax_sec = climax_frame / fps
                                extraction_method = "optical_flow_peak+vlm_refinement"
                                vlm_confidence = 1.0
            except Exception as e:
                print(f"[climax_extraction] VLM refinement failed: {e}")

        velocity = task.get('task_velocity', 'medium')
        window = _reaction_window(climax_sec, velocity, duration_sec)

        meta = {
            "task_climax_sec": round(climax_sec, 2),
            "task_reaction_window_sec": window,
            "climax_extraction_method": extraction_method,
            "optical_flow_peak_magnitude": round(float(max_flow), 2),
        }
        if vlm_confidence is not None:
            meta["vlm_climax_confidence"] = vlm_confidence
        task['task_temporal_metadata'] = meta


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
    if workers > 1 and len(args) > 1:
        with Pool(processes=min(workers, len(args))) as pool:
            for idx, (entry, was_updated) in zip(todo, pool.map(_populate_one_entry, args)):
                entries[idx] = entry
                updated += 1 if was_updated else 0
    else:
        for idx, a in zip(todo, args):
            entry, was_updated = _populate_one_entry(a)
            entries[idx] = entry
            updated += 1 if was_updated else 0

    if updated:
        with open(manifest_path, 'w') as f:
            json.dump(entries, f, indent=4)
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
