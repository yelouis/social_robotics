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
from src.layer_05_visualizer.picker import pick_video
from src.layer_05_visualizer.hydrate import build_overlay_bundle
from src.layer_05_visualizer.render import render_clip
from src.config import DATASET_PATHS

def main():
    parser = argparse.ArgumentParser(description="Layer 05 Signal Visualizer")
    parser.add_argument("--video-id", type=str, help="Skip picker, render this clip directly")
    parser.add_argument("--video", type=str, help="Select by file path instead of ID")
    parser.add_argument("--list", action="store_true", help="Print catalog table and exit")
    parser.add_argument("--json", action="store_true", help="Format --list output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Detailed --list output")
    parser.add_argument("--all", action="store_true", help="Batch render all discovered clips")
    
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
        
    if args.all:
        catalog = build_catalog(scan_roots)
        entries_to_render = list(catalog.values())
    elif args.video_id:
        entries_to_render = [resolve_video(args.video_id, scan_roots)]
    elif args.video:
        vid_id = Path(args.video).stem
        entries_to_render = [resolve_video(vid_id, scan_roots)]
    else:
        # Interactive mode
        catalog = build_catalog(scan_roots)
        try:
            chosen = pick_video(catalog)
            if not chosen:
                print("No clip selected, exiting.")
                return
            entries_to_render = [chosen]
        except Exception as e:
            # Catch TclError or similar if no display
            print("GUI unavailable. Falling back to --list.")
            print(list_catalog(catalog, as_json=args.json, verbose=args.verbose))
            print("\nPlease run with --video-id <uuid> to render.")
            sys.exit(1)
            
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
            
        print(f"Hydrating bundle for {entry.video_id}...")
        bundle = build_overlay_bundle(entry, include_phantoms=args.show_phantoms)
        
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
