#!/usr/bin/env python3
"""
CLI entry point for the Layer 05 Visualizer.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer_05_visualizer.catalog import build_catalog, list_catalog, resolve_video
from src.layer_05_visualizer.picker import pick_video, pick_video_terminal
from src.layer_05_visualizer.hydrate import build_overlay_bundle
from src.layer_05_visualizer.render import render_clip
from src.config import DATASET_PATHS

def verify_visualizer(scan_roots, out_dir, video_id=None) -> bool:
    """Self-test the whole pipeline on one real video and report PASS/FAIL.

    Discovers a clip (the one given, or the best-findings clip whose video file is
    present), renders a 2-second downscaled annotated clip, and asserts the output
    is a real video with frames and that overlays were actually drawn onto it.
    """
    import cv2
    import numpy as np

    print("=== Visualizer self-check (--verify) ===")
    catalog = build_catalog(scan_roots)
    print(f"[1/5] Catalog discovery: {len(catalog)} clips found")
    if not catalog:
        print("FAIL: no clips discovered. Check --scan-root.")
        return False

    if video_id:
        entry = catalog.get(video_id) or resolve_video(video_id, scan_roots)
    else:
        avail = [e for e in catalog.values() if e.video_path and Path(e.video_path).exists()]
        avail.sort(key=lambda e: e.num_layers_with_findings, reverse=True)
        if not avail:
            print("FAIL: no clip has a local video file (is the Extreme SSD mounted?).")
            return False
        entry = avail[0]
    print(f"[2/5] Target clip: {entry.video_id}  (★{entry.num_layers_with_findings} findings)")

    if not entry.video_path or not Path(entry.video_path).exists():
        print(f"FAIL: video file missing for {entry.video_id}: {entry.video_path}")
        return False

    bundle = build_overlay_bundle(entry)
    n_boxes = sum(len(t["boxes"]) for t in bundle["tracks"])
    n_windows = sum(len(t["windows"]) for t in bundle["tracks"])
    print(f"[3/5] Hydrated bundle: {len(bundle['tracks'])} tracks, {n_boxes} boxes, {n_windows} windows")
    if n_boxes == 0:
        print("FAIL: bundle has zero bystander boxes — nothing would be drawn.")
        return False

    out_path = Path(out_dir) / f"_verify_{entry.video_id[:8]}.mp4"
    print(f"[4/5] Rendering 2s smoke clip -> {out_path}")
    render_clip(bundle=bundle, video_path=Path(entry.video_path), out_path=out_path,
                scale=0.5, layers="all", panels=["timeline", "readout"],
                with_audio=False, clip_range=(0.0, 2.0))

    cap = cv2.VideoCapture(str(out_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok_frame, frame = cap.read()
    cap.release()
    if not out_path.exists() or n_frames <= 0 or not ok_frame:
        print(f"FAIL: output not a valid video (frames={n_frames}).")
        return False
    # Compare an annotated frame to the raw source frame: overlays must change pixels.
    src = cv2.VideoCapture(str(entry.video_path))
    _, raw = src.read()
    src.release()
    raw = cv2.resize(raw, (frame.shape[1], frame.shape[0])) if raw is not None else None
    changed = int((frame != raw).any(axis=2).sum()) if raw is not None else -1
    print(f"[5/5] Output OK: {n_frames} frames, {frame.shape[1]}x{frame.shape[0]}, "
          f"overlay pixels changed vs source: {changed}")
    if changed == 0:
        print("FAIL: annotated frame is identical to source — no overlays drawn.")
        return False

    print(f"\nPASS ✓  The visualizer works. Open: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Layer 05 Signal Visualizer")
    parser.add_argument("--video-id", type=str, help="Skip picker, render this clip directly")
    parser.add_argument("--video", type=str, help="Select by file path instead of ID")
    parser.add_argument("--list", action="store_true", help="Print catalog table and exit")
    parser.add_argument("--json", action="store_true", help="Format --list output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Detailed --list output")
    parser.add_argument("--all", action="store_true", help="Batch render all discovered clips")
    parser.add_argument("--gui", action="store_true", help="Use the Tkinter window picker (may be blank on the macOS system Tk 8.5; terminal picker is the default)")
    parser.add_argument("--verify", action="store_true", help="Self-test: render a 2s clip for one video and report PASS/FAIL")
    
    parser.add_argument("--scan-root", type=str, action="append", default=[], help="Additional directories to scan")
    parser.add_argument("--out-dir", type=str, default="e2e_reports/viz", help="Output directory")
    
    parser.add_argument("--layers", type=str, default="all", help="Comma-separated layers or 'all'")
    parser.add_argument("--people", type=str, help="Comma-separated person IDs to render")
    parser.add_argument("--show-phantoms", action="store_true", help="Draw negative-ID phantom tracks")
    parser.add_argument("--panels", type=str, default="timeline,readout", help="Comma-separated panels or 'none'")
    parser.add_argument("--scale", type=float, default=1.0, help="Downscale output (e.g. 0.5)")
    
    # Audio args
    parser.add_argument("--with-audio", action="store_true", default=True, help="Mux original audio (default)")
    parser.add_argument("--no-audio", action="store_false", dest="with_audio", help="Do not mux audio")
    
    parser.add_argument("--clip-range", type=str, help="Render range e.g. 40:60")
    parser.add_argument("--force", action="store_true", help="Re-render even if output exists")
    parser.add_argument("--dense-boxes", action="store_true", default=True, help="Re-detect bystanders densely in reaction windows so boxes follow the subject (default)")
    parser.add_argument("--no-dense-boxes", action="store_false", dest="dense_boxes", help="Use Node-02's sparse boxes as-is (boxes freeze between detections)")
    
    args = parser.parse_args()
    
    scan_roots = [Path("e2e_reports")]
    # Add Ego4D configured paths
    if "ego4d" in DATASET_PATHS:
        scan_roots.extend(DATASET_PATHS["ego4d"])
    scan_roots.extend([Path(r) for r in args.scan_root])
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    clip_range = None
    if args.clip_range:
        parts = args.clip_range.split(":")
        clip_range = (float(parts[0]), float(parts[1]))
        
    people = None
    if args.people:
        people = args.people.split(",")
        
    panels = []
    if args.panels and args.panels.lower() != "none":
        panels = args.panels.split(",")
    
    if args.list:
        catalog = build_catalog(scan_roots)
        print(list_catalog(catalog, as_json=args.json, verbose=args.verbose))
        return

    if args.verify:
        ok = verify_visualizer(scan_roots, out_dir, video_id=args.video_id)
        sys.exit(0 if ok else 1)

    if args.all:
        catalog = build_catalog(scan_roots)
        entries_to_render = list(catalog.values())
    elif args.video_id:
        entries_to_render = [resolve_video(args.video_id, scan_roots)]
    elif args.video:
        vid_id = Path(args.video).stem
        entries_to_render = [resolve_video(vid_id, scan_roots)]
    else:
        # Interactive selection. Terminal picker is the default (always works);
        # --gui opts into the Tkinter window (may be blank on macOS system Tk 8.5).
        catalog = build_catalog(scan_roots)
        chosen = None
        if args.gui:
            try:
                chosen = pick_video(catalog)
            except Exception as e:
                print(f"GUI picker unavailable ({e}); using the terminal picker instead.")
                chosen = pick_video_terminal(catalog)
        else:
            chosen = pick_video_terminal(catalog)
        if not chosen:
            print("No clip selected, exiting.")
            return
        entries_to_render = [chosen]
            
    for entry in entries_to_render:
        # Construct output name
        l_str = f"L-{args.layers}"
        s_str = f"s{int(args.scale * 100)}" if args.scale != 1.0 else ""
        rng_str = f"_{args.clip_range.replace(':', '-')}" if args.clip_range else ""
        
        parts = filter(None, [entry.video_id, l_str, s_str, rng_str])
        out_name = "__".join(parts) + ".mp4"
        out_path = out_dir / out_name
        
        if out_path.exists() and not args.force:
            print(f"Skipping {entry.video_id} - output exists at {out_path} (use --force to overwrite)")
            continue
            
        print(f"Hydrating bundle for {entry.video_id}..." + (" (dense boxes)" if args.dense_boxes else ""))
        bundle = build_overlay_bundle(entry, include_phantoms=args.show_phantoms, dense_boxes=args.dense_boxes)
        
        print(f"Rendering {entry.video_id} -> {out_path}...")
        if not entry.video_path or not entry.video_path.exists():
            print(f"Error: Missing video file for {entry.video_id}")
            continue
            
        try:
            render_clip(
                bundle=bundle,
                video_path=entry.video_path,
                out_path=out_path,
                scale=args.scale,
                layers=args.layers,
                people=people,
                show_phantoms=args.show_phantoms,
                panels=panels,
                with_audio=args.with_audio,
                clip_range=clip_range
            )
            print(f"Done: {out_path}")
        except Exception as e:
            print(f"Render failed for {entry.video_id}: {e}")

if __name__ == "__main__":
    main()
