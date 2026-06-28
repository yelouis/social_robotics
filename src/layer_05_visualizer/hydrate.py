"""
Hydration step. Merges the manifest and all layer results into a single Overlay Bundle JSON.
"""
import cv2
import json
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
) -> dict:
    
    native_w, native_h, fps, has_audio = 1920, 1080, 30.0, True
    if probe_video and entry.video_path and entry.video_path.exists():
        cap = cv2.VideoCapture(str(entry.video_path))
        if cap.isOpened():
            native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            # Accurate audio probe would require ffprobe/ffmpeg, but we'll fallback to manifest
            has_audio = entry.manifest_entry.get("has_audio", True)
        cap.release()
    else:
        native_w = entry.manifest_entry.get("video_width", 1920)
        native_h = entry.manifest_entry.get("video_height", 1080)
        fps = entry.manifest_entry.get("fps", 30.0)
        has_audio = entry.manifest_entry.get("has_audio", True)
        
    people_map = {}
    tracks_by_id = {}
    
    # 02 Manifest - Bystander boxes
    for det in entry.manifest_entry.get("bystander_detections", []):
        t = det.get("timestamp_sec", 0.0)
        for t_box in det.get("tracked_boxes", []):
            pid = t_box.get("person_id", -1)
            if not include_phantoms and pid < 0:
                continue
                
            if pid not in people_map:
                people_map[pid] = {
                    "color": get_person_color_hex(pid),
                    "phantom": pid < 0
                }
                tracks_by_id[pid] = {
                    "person_id": pid,
                    "boxes": [],
                    "gap_tolerance_sec": 4.0, # default D2
                    "windows": [],
                    "attention": None
                }
                
            b = t_box.get("box_2d", [0,0,0,0])
            if native_w and native_h:
                # normalize
                nx1 = max(0.0, min(1.0, b[0] / native_w))
                ny1 = max(0.0, min(1.0, b[1] / native_h))
                nx2 = max(0.0, min(1.0, b[2] / native_w))
                ny2 = max(0.0, min(1.0, b[3] / native_h))
                
                tracks_by_id[pid]["boxes"].append({
                    "t": t,
                    "box": [nx1, ny1, nx2, ny2],
                    "conf": t_box.get("confidence", 1.0)
                })

    # Sort boxes
    for trk in tracks_by_id.values():
        trk["boxes"].sort(key=lambda x: x["t"])
        
    # Layer 03a Attention
    if "03a" in entry.results_by_layer:
        res = entry.results_by_layer["03a"]
        for p in res.get("people", []):
            pid = p.get("person_id")
            if pid in tracks_by_id:
                # simplify trace loading for demo
                tracks_by_id[pid]["attention"] = {
                    "summary": p.get("summary", {}),
                    "trace": sorted(p.get("trace", []), key=lambda x: x["t"])
                }
                
    # Layer 03b Emotion
    if "03b" in entry.results_by_layer:
        res = entry.results_by_layer["03b"]
        for s in res.get("emotion_slices", []):
            pid = s.get("person_id")
            if pid in tracks_by_id:
                w = s.copy()
                w["layer"] = "03b"
                w["kind"] = "emotion_slice"
                tracks_by_id[pid]["windows"].append(w)
                
    # Layer 03d Proxemic
    if "03d" in entry.results_by_layer:
        res = entry.results_by_layer["03d"]
        for p in res.get("people", []):
            pid = p.get("person_id")
            if pid in tracks_by_id:
                w = p.copy()
                w["layer"] = "03d"
                w["kind"] = "proxemic"
                tracks_by_id[pid]["windows"].append(w)

    # Layer 03e Gesture
    if "03e" in entry.results_by_layer:
        res = entry.results_by_layer["03e"]
        for p in res.get("people", []):
            pid = p.get("person_id")
            if pid in tracks_by_id:
                w = p.copy()
                w["layer"] = "03e"
                w["kind"] = "gesture"
                tracks_by_id[pid]["windows"].append(w)

    # Layer 03f Motor
    if "03f" in entry.results_by_layer:
        res = entry.results_by_layer["03f"]
        for p in res.get("people", []):
            pid = p.get("person_id")
            if pid in tracks_by_id:
                w = p.copy()
                w["layer"] = "03f"
                w["kind"] = "motor_resonance"
                tracks_by_id[pid]["windows"].append(w)
                
    # Tasks
    tasks = []
    # 03f tasks carry ego_kinetic_chaos_score
    ego_scores = {}
    if "03f" in entry.results_by_layer:
        for t in entry.results_by_layer["03f"].get("tasks", []):
            ego_scores[t.get("task_id")] = t.get("ego_kinetic_chaos_score", 0.0)
            
    for t in entry.manifest_entry.get("identified_tasks", []):
        t_out = t.copy()
        t_id = t.get("task_id")
        if t_id in ego_scores:
            t_out["ego_kinetic_chaos_score"] = ego_scores[t_id]
        tasks.append(t_out)
        
    # Audio (03c)
    audio = []
    if "03c" in entry.results_by_layer:
        audio = entry.results_by_layer["03c"].get("tasks", [])
        
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "video_id": entry.video_id,
        "source_dataset": entry.manifest_entry.get("source_dataset", "ego4d"),
        "clip": {
            "video_path": str(entry.video_path) if entry.video_path else None,
            "native_width": native_w,
            "native_height": native_h,
            "fps": fps,
            "duration_sec": entry.manifest_entry.get("duration_sec", 0.0),
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
