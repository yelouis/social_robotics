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
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional

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

        for frame_idx in range(start_frame + step, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
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
            for frame_idx in range(dense_start + 1, dense_end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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


def populate_climax_for_manifest(
    manifest_path: Path,
    vlm_model: Optional[str] = None,
    skip_vlm: bool = False,
) -> int:
    """Fill in `task_temporal_metadata` for every entry in `manifest_path`
    that has tasks with empty metadata. Writes back to the same path. Returns
    the number of entries updated. Idempotent — subsequent calls are no-ops
    once every task has metadata.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return 0
    with open(manifest_path, 'r') as f:
        entries = json.load(f)

    updated = 0
    for entry in entries:
        tasks = entry.get('identified_tasks', [])
        if not tasks or all(t.get('task_temporal_metadata') for t in tasks):
            continue
        video_path = entry.get('video_path')
        if not video_path or not Path(video_path).exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or entry.get('fps') or 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = entry.get('duration_sec') or (total_frames / fps if fps else 0.0)
            if fps <= 0 or duration_sec <= 0:
                continue
            compute_task_climax_for_video(
                cap, fps, total_frames, tasks, duration_sec,
                vlm_model=vlm_model, skip_vlm=skip_vlm,
            )
            updated += 1
        finally:
            cap.release()

    if updated:
        with open(manifest_path, 'w') as f:
            json.dump(entries, f, indent=4)
    return updated
