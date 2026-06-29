"""
Hydration step. Merges the manifest and all layer results into a single Overlay Bundle JSON.
"""
import cv2
import json
from dataclasses import replace
from pathlib import Path
from typing import Union, Optional

from src.layer_05_visualizer.bundle_schema import SCHEMA_VERSION
from src.layer_05_visualizer.catalog import CatalogEntry, resolve_video
from src.layer_05_visualizer.colors import get_person_color_hex

def build_overlay_bundle(
    entry: CatalogEntry,
    *,
    probe_video: bool = True,
    include_phantoms: bool = True,
    dense_boxes: bool = False,
) -> dict:

    # --dense-boxes: re-detect the bystanders densely inside the reaction windows
    # so the boxes track the moving subject (Node-02's ~3-6s boxes otherwise freeze
    # between detections). Matched back to the manifest person_ids, so the layer
    # signals stay attached. See src/shared/dense_detect.py.
    if dense_boxes and entry.video_path and Path(entry.video_path).exists():
        try:
            from src.shared.dense_detect import densify_manifest_entry
            dense_entry, _stats = densify_manifest_entry(entry.manifest_entry)
            entry = replace(entry, manifest_entry=dense_entry)
        except Exception as e:
            print(f"[hydrate] dense-boxes skipped ({e}); using sparse manifest boxes.")

    native_w, native_h, fps = 1920, 1080, 30.0
    # Audio truth comes from the catalog (03c prosody). None = unknown -> stay optimistic
    # so ffmpeg attempts the mux; render_clip falls back to silent if there is no track.
    has_audio = entry.has_audio if entry.has_audio is not None else True
    if probe_video and entry.video_path and entry.video_path.exists():
        cap = cv2.VideoCapture(str(entry.video_path))
        if cap.isOpened():
            native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or native_w
            native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or native_h
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or fps
        cap.release()
    else:
        native_w = int(entry.manifest_entry.get("video_width", 1920) or 1920)
        native_h = int(entry.manifest_entry.get("video_height", 1080) or 1080)
        try:
            fps = float(entry.manifest_entry.get("fps", 30.0))
        except (TypeError, ValueError):
            fps = 30.0
        
    people_map = {}
    tracks_by_id = {}

    def _f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _ensure_track(pid):
        if pid not in people_map:
            people_map[pid] = {"color": get_person_color_hex(pid), "phantom": pid < 0}
            tracks_by_id[pid] = {
                "person_id": pid,
                "boxes": [],
                "gap_tolerance_sec": 4.0,  # default D2
                "windows": [],
                "attention": None,
            }

    # 02 Manifest - Bystander boxes.
    # Real schema: bystander_detections is a list of PER-PERSON records, each with
    # parallel arrays timestamps_sec[] / bounding_boxes[] / detection_confidence[];
    # each box is [x1, y1, x2, y2] in NATIVE pixels.
    for det in entry.manifest_entry.get("bystander_detections", []):
        pid = det.get("person_id", -1)
        if not include_phantoms and pid < 0:
            continue
        timestamps = det.get("timestamps_sec", []) or []
        boxes = det.get("bounding_boxes", []) or []
        confs = det.get("detection_confidence", []) or []
        if not timestamps or not boxes or not (native_w and native_h):
            continue
        _ensure_track(pid)
        for i, ts in enumerate(timestamps):
            if i >= len(boxes):
                break
            b = boxes[i]
            if not b or len(b) != 4:
                continue
            nx1 = max(0.0, min(1.0, _f(b[0]) / native_w))
            ny1 = max(0.0, min(1.0, _f(b[1]) / native_h))
            nx2 = max(0.0, min(1.0, _f(b[2]) / native_w))
            ny2 = max(0.0, min(1.0, _f(b[3]) / native_h))
            tracks_by_id[pid]["boxes"].append({
                "t": _f(ts),
                "box": [nx1, ny1, nx2, ny2],
                "conf": _f(confs[i], 1.0) if i < len(confs) else 1.0,
            })

    for trk in tracks_by_id.values():
        trk["boxes"].sort(key=lambda x: x["t"])

    # Map manifest task_id -> (climax_sec, reaction_window) for layers whose
    # per-person verdict has no window of its own (03f) and for the task timeline.
    task_climax = {}
    task_window = {}
    for t in entry.manifest_entry.get("identified_tasks", []):
        tm = t.get("task_temporal_metadata", {}) or {}
        tid = t.get("task_id")
        task_climax[tid] = _f(tm.get("task_climax_sec", 0.0))
        rwin = tm.get("task_reaction_window_sec")
        if rwin and len(rwin) == 2:
            task_window[tid] = [_f(rwin[0]), _f(rwin[1])]

    def _iter_per_person(res):
        for task in res.get("tasks_analyzed", []) or []:
            for p in task.get("per_person", []) or []:
                yield task, p

    # Layer 03a Attention — per_person[] with inline summary fields + attention_trace[]
    if "03a" in entry.results_by_layer:
        for p in entry.results_by_layer["03a"].get("per_person", []) or []:
            pid = p.get("person_id")
            if pid in tracks_by_id:
                summary = {k: v for k, v in p.items() if k != "attention_trace"}
                trace = sorted(p.get("attention_trace", []) or [], key=lambda x: _f(x.get("t")))
                tracks_by_id[pid]["attention"] = {"summary": summary, "trace": trace}

    # Layer 03b Emotion — tasks_analyzed[].per_person[].temporal_slices[] (each slice is a window)
    if "03b" in entry.results_by_layer:
        for task, p in _iter_per_person(entry.results_by_layer["03b"]):
            pid = p.get("person_id")
            if pid not in tracks_by_id:
                continue
            for s in p.get("temporal_slices", []) or []:
                tracks_by_id[pid]["windows"].append({
                    "layer": "03b", "kind": "emotion_slice",
                    "window_sec": s.get("window_sec"),
                    "transition_pair": s.get("transition_pair"),
                    "classified_direction": s.get("classified_direction"),
                    "terminal_magnitude": s.get("terminal_magnitude"),
                })

    # Layer 03d Proxemic — per_person[].measurement_window_sec
    if "03d" in entry.results_by_layer:
        for task, p in _iter_per_person(entry.results_by_layer["03d"]):
            pid = p.get("person_id")
            if pid in tracks_by_id:
                w = dict(p)
                w["layer"] = "03d"; w["kind"] = "proxemic"
                w["window_sec"] = p.get("measurement_window_sec")
                tracks_by_id[pid]["windows"].append(w)

    # Layer 03e Gesture — per_person[].measurement_window_sec
    if "03e" in entry.results_by_layer:
        for task, p in _iter_per_person(entry.results_by_layer["03e"]):
            pid = p.get("person_id")
            if pid in tracks_by_id:
                w = dict(p)
                w["layer"] = "03e"; w["kind"] = "gesture"
                w["window_sec"] = p.get("measurement_window_sec")
                tracks_by_id[pid]["windows"].append(w)

    # Layer 03f Motor — per_person[] has NO window; anchor to the task reaction window
    if "03f" in entry.results_by_layer:
        for task, p in _iter_per_person(entry.results_by_layer["03f"]):
            pid = p.get("person_id")
            if pid in tracks_by_id:
                w = dict(p)
                w["layer"] = "03f"; w["kind"] = "motor_resonance"
                w["window_sec"] = task_window.get(task.get("task_id"))
                tracks_by_id[pid]["windows"].append(w)

    # Tasks (timeline). 03f tasks_analyzed[] carry ego_kinetic_chaos_score.
    ego_scores = {}
    if "03f" in entry.results_by_layer:
        for t in entry.results_by_layer["03f"].get("tasks_analyzed", []) or []:
            ego_scores[t.get("task_id")] = _f(t.get("ego_kinetic_chaos_score", 0.0))

    tasks = []
    for t in entry.manifest_entry.get("identified_tasks", []):
        tid = t.get("task_id")
        rwin = task_window.get(tid)
        t_out = {
            "task_id": tid,
            "task_label": " ".join(str(t.get("task_label", "")).split()),
            "task_velocity": t.get("task_velocity"),
            "climax_sec": task_climax.get(tid, 0.0),
            "reaction_windows_sec": [rwin] if rwin else [],
        }
        if tid in ego_scores:
            t_out["ego_kinetic_chaos_score"] = ego_scores[tid]
        tasks.append(t_out)

    # Audio (03c) — tasks_analyzed[] (carried for completeness; not drawn by the current renderer)
    audio = []
    if "03c" in entry.results_by_layer:
        audio = entry.results_by_layer["03c"].get("tasks_analyzed", []) or []
        
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "video_id": entry.video_id,
        "source_dataset": entry.manifest_entry.get("source_dataset", "ego4d"),
        "clip": {
            "video_path": str(entry.video_path) if entry.video_path else None,
            "native_width": native_w,
            "native_height": native_h,
            "fps": fps,
            "duration_sec": _f(entry.manifest_entry.get("duration_sec", 0.0)),
            "has_audio": has_audio
        },
        "layers_present": ["02_manifest"] + list(entry.results_by_layer.keys()),
        "people": {str(k): v for k, v in people_map.items()},
        "tracks": list(tracks_by_id.values()),
        "tasks": tasks,
        "audio": audio,
        "audio_envelope": None,
        "hands": []
    }
    
    return bundle

def build_bundle_for_video(video_id: str, scan_roots: list[Union[str, Path]], **kw) -> dict:
    entry = resolve_video(video_id, scan_roots)
    return build_overlay_bundle(entry, **kw)

def write_bundles_for_catalog(scan_roots, out_dir, *, video_ids: Optional[list[str]] = None) -> list[Path]:
    from src.layer_05_visualizer.catalog import build_catalog
    catalog = build_catalog(scan_roots)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    if video_ids:
        to_process = {vid: catalog[vid] for vid in video_ids if vid in catalog}
    else:
        to_process = catalog
        
    paths = []
    for vid, entry in to_process.items():
        bundle = build_overlay_bundle(entry)
        p = out / f"{vid}.bundle.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        paths.append(p)
        
    return paths
