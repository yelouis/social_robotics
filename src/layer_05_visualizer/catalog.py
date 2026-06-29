"""
Auto-discovery and indexing for the Visualizer.
Finds manifests, layer results, and videos, stitching them by video_id.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from src.config import DATASET_PATHS, BASE_DIR

@dataclass
class CatalogEntry:
    video_id: str
    manifest_entry: dict
    results_by_layer: dict[str, dict]
    video_path: Optional[Path]
    findings: dict[str, str]
    num_layers_with_findings: int
    summary_text: str
    sources: dict
    has_audio: Optional[bool] = None   # True/False from 03c prosody, None = unknown (03c absent)

# Finding Predicates.
#
# Every layer-03 result nests its data under `tasks_analyzed[]`; per-bystander
# verdicts live in `tasks_analyzed[].per_person[]` (03c is the exception — its
# signal is on the task itself under `prosody_metrics`). A "finding" means the
# layer actually detected a social signal, as opposed to running but reporting
# nothing. The paths below mirror the schemas the renderer draws, so the
# picker's star count can never disagree with the burned overlays.
def _tasks(res: dict) -> list:
    return res.get("tasks_analyzed", []) or []

def _all_people(res: dict):
    for t in _tasks(res):
        for p in t.get("per_person", []) or []:
            yield p

def _finding_03a(res: dict) -> bool:
    return bool(res.get("aggregate", {}).get("any_person_engaged", False))

def _finding_03b(res: dict) -> bool:
    for p in _all_people(res):
        for s in p.get("temporal_slices", []) or []:
            if s.get("classified_direction", "neutral") != "neutral":
                return True
    return False

def _finding_03c(res: dict) -> bool:
    for t in _tasks(res):
        pm = t.get("prosody_metrics", {}) or {}
        if pm.get("audio_present", False):
            if t.get("classified_acoustic_tone", "Neutral") != "Neutral" or pm.get("dominant_emotion", "neutral") != "neutral":
                return True
    return False

def _finding_03d(res: dict) -> bool:
    return any(p.get("classified_action", "Neutral") != "Neutral" for p in _all_people(res))

def _finding_03e(res: dict) -> bool:
    return any(p.get("gesture_detected", "none") != "none" for p in _all_people(res))

def _finding_03f(res: dict) -> bool:
    return any(p.get("motor_resonance_detected", False) or p.get("mirroring_detected", False) for p in _all_people(res))

LAYER_FINDING_PREDICATES: dict[str, Callable[[dict], bool]] = {
    "03a": _finding_03a,
    "03b": _finding_03b,
    "03c": _finding_03c,
    "03d": _finding_03d,
    "03e": _finding_03e,
    "03f": _finding_03f,
}

def _parse_tasks(manifest_entry: dict) -> list[dict]:
    # Try different manifest formats for task windows
    return manifest_entry.get("identified_tasks", [])

def _audio_from_03c(results: dict) -> Optional[bool]:
    # The manifest has no audio flag; the only honest source is 03c's prosody.
    # None = 03c didn't run, so we genuinely don't know.
    res = results.get("03c")
    if not res:
        return None
    for t in res.get("tasks_analyzed", []) or []:
        if (t.get("prosody_metrics", {}) or {}).get("audio_present", False):
            return True
    return False

def _coerce_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _build_summary_text(video_id: str, manifest_entry: dict, duration_sec: float, has_audio: Optional[bool], findings: dict[str, str], layers_found: list[str]) -> str:
    lines = []
    audio_str = {True: "yes", False: "no", None: "unknown"}[has_audio]
    dataset = manifest_entry.get("source_dataset", "ego4d")
    lines.append(f"{video_id}  {dataset} · {duration_sec:.1f}s · audio:{audio_str}")

    tasks = _parse_tasks(manifest_entry)
    for t in tasks:
        # The manifest nests timing under task_temporal_metadata.
        tm = t.get("task_temporal_metadata", {}) or {}
        climax = _coerce_float(tm.get("task_climax_sec", 0))
        window = tm.get("task_reaction_window_sec")
        label = " ".join(str(t.get("task_label", "unknown")).split())  # collapse newlines
        window_str = f" · reaction {window}" if window else ""
        lines.append(f"TASK {label}   climax {climax:.1f}s{window_str}")

    if not tasks:
        lines.append("TASK no tasks found")

    finding_parts = []
    for layer in ["03a", "03b", "03c", "03d", "03e", "03f"]:
        st = findings.get(layer, "absent")
        if st == "finding":
            finding_parts.append(f"{layer} finding")
        elif st == "ran":
            finding_parts.append(f"{layer} ran - no finding")
        else:
            finding_parts.append(f"{layer} absent")
            
    # break into two lines
    if len(finding_parts) >= 3:
        lines.append(" · ".join(finding_parts[:3]))
        lines.append(" · ".join(finding_parts[3:]))
    else:
        lines.append(" · ".join(finding_parts))

    return "\n".join(lines)

def build_catalog(scan_roots: list[Union[str, Path]]) -> dict[str, CatalogEntry]:
    catalog = {}
    
    # Internal structures for recency
    manifests_by_vid = {} # vid -> (mtime, entry, path)
    results_by_layer_vid = {} # vid -> {layer: (mtime, entry, path)}
    video_paths = {} # vid -> path
    
    for root in scan_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        
        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
                
            if path.suffix == ".mp4":
                # Only use uuid-like files
                vid = path.stem
                # Keep first found
                if vid not in video_paths:
                    video_paths[vid] = path
            
            elif path.name.endswith(".json"):
                mtime = path.stat().st_mtime
                if "manifest" in path.name.lower():
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for entry in data:
                                    if "video_id" in entry:
                                        vid = entry["video_id"]
                                        if vid not in manifests_by_vid or mtime > manifests_by_vid[vid][0]:
                                            manifests_by_vid[vid] = (mtime, entry, path)
                    except Exception:
                        pass
                
                elif "03" in path.name and "_result.json" in path.name:
                    # e2e_reports/.../03a_result.json
                    layer = path.name.split("_")[0] # e2e.g. 03a
                    if layer in LAYER_FINDING_PREDICATES:
                        try:
                            with open(path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for entry in data:
                                        if "video_id" in entry:
                                            vid = entry["video_id"]
                                            if vid not in results_by_layer_vid:
                                                results_by_layer_vid[vid] = {}
                                            if layer not in results_by_layer_vid[vid] or mtime > results_by_layer_vid[vid][layer][0]:
                                                results_by_layer_vid[vid][layer] = (mtime, entry, path)
                        except Exception:
                            pass
                            
    # Build entries
    all_vids = set(manifests_by_vid.keys()) | set(results_by_layer_vid.keys())
    
    for vid in all_vids:
        # Require manifest
        if vid not in manifests_by_vid:
            continue
            
        manifest_entry = manifests_by_vid[vid][1]
        manifest_path = manifests_by_vid[vid][2]
        
        results = {}
        sources = {"manifest": str(manifest_path)}
        
        findings = {}
        num_findings = 0
        
        layer_results = results_by_layer_vid.get(vid, {})
        for layer, (mtime, entry, path) in layer_results.items():
            results[layer] = entry
            sources[layer] = str(path)
            
            # evaluate findings
            pred = LAYER_FINDING_PREDICATES[layer]
            if pred(entry):
                findings[layer] = "finding"
                num_findings += 1
            else:
                findings[layer] = "ran"
                
        for layer in LAYER_FINDING_PREDICATES:
            if layer not in findings:
                findings[layer] = "absent"
                
        vpath = video_paths.get(vid)
        if not vpath and "video_path" in manifest_entry:
            cand = Path(manifest_entry["video_path"])
            if cand.exists():
                vpath = cand
                
        dur = float(manifest_entry.get("duration_sec", 0.0) or 0.0)
        has_audio = _audio_from_03c(results)   # True / False / None(unknown)

        summary = _build_summary_text(vid, manifest_entry, dur, has_audio, findings, list(results.keys()))

        catalog[vid] = CatalogEntry(
            video_id=vid,
            manifest_entry=manifest_entry,
            results_by_layer=results,
            video_path=vpath,
            findings=findings,
            num_layers_with_findings=num_findings,
            summary_text=summary,
            sources=sources,
            has_audio=has_audio,
        )
        
    return catalog

def list_catalog(catalog: dict[str, CatalogEntry], *, as_json=False, verbose=False) -> str:
    if as_json:
        # not fully implemented, just dump basic info
        return json.dumps({v.video_id: v.summary_text for v in catalog.values()}, indent=2)
        
    lines = []
    # Sort by stars desc
    sorted_vids = sorted(catalog.values(), key=lambda c: c.num_layers_with_findings, reverse=True)
    
    header = f"{'★':<2} | {'task':<25} | {'layers':<15} | {'audio':<5} | {'video':<5} | {'id':<8}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for c in sorted_vids:
        star = str(c.num_layers_with_findings)
        tasks = _parse_tasks(c.manifest_entry)
        task_label = " ".join(str(tasks[0].get("task_label", "unknown")).split()) if tasks else "none"
        if len(task_label) > 25:
            task_label = task_label[:22] + "..."

        finding_layers = sorted(l for l, st in c.findings.items() if st == "finding")
        layers_str = " ".join(finding_layers) if finding_layers else "-"
        if len(layers_str) > 15:
            layers_str = layers_str[:12] + "..."

        audio_str = {True: "yes", False: "no", None: "?"}[c.has_audio]
        vid_str = "y" if c.video_path else "n"
        short_id = c.video_id[:8]
        
        row = f"{star:<2} | {task_label:<25} | {layers_str:<15} | {audio_str:<5} | {vid_str:<5} | {short_id:<8}"
        lines.append(row)
        if verbose:
            lines.append("  Summary:")
            for s_line in c.summary_text.splitlines():
                lines.append(f"    {s_line}")
                
    return "\n".join(lines)

def resolve_video(video_id: str, scan_roots: list[Union[str, Path]]) -> CatalogEntry:
    catalog = build_catalog(scan_roots)
    if video_id not in catalog:
        raise ValueError(f"video_id {video_id} not found under any --scan-root")
    
    entry = catalog[video_id]
    if not entry.video_path:
        raise ValueError(f"video file for {video_id} not found")
        
    return entry
