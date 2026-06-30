#!/usr/bin/env python3
"""
Window-bounded bystander densification pre-pass (docs/02 Issue 1 → Option A).

Reads a Node-02 manifest and emits a *densified* copy in which every clip's
bystander boxes are re-detected densely (YOLO+ByteTrack, ~10 fps) **inside the
task reaction windows** and IoU-matched back to the manifest person_ids
(src/shared/dense_detect.py). Node-02 itself is left untouched; the kinematic
layers (03d/03f) are simply pointed at the dense manifest instead of the sparse
one.

    python tools/densify_manifest.py --in filtered_manifest.json \
                                     --out filtered_manifest.dense.json

Resumable: clips already present in --out (by video_id) are skipped, so a long
corpus pass can be re-run after an interruption.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.dense_detect import densify_manifest_entry, _DenseDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="source Node-02 manifest (list of entries)")
    ap.add_argument("--out", required=True, help="destination densified manifest")
    ap.add_argument("--fps-target", type=float, default=10.0)
    ap.add_argument("--pad-sec", type=float, default=4.0, help="pad each reaction window by this many seconds")
    ap.add_argument("--force", action="store_true", help="re-densify clips already in --out")
    a = ap.parse_args()

    src = json.loads(Path(a.inp).read_text())
    out_path = Path(a.out)
    done = {}
    if out_path.exists() and not a.force:
        try:
            done = {e.get("video_id", e.get("id")): e for e in json.loads(out_path.read_text())}
            print(f"[densify] resuming: {len(done)} clips already in {out_path.name}")
        except Exception:
            done = {}

    detector = _DenseDetector()
    results = list(done.values())
    totals = {"clips": 0, "densified": 0, "added": 0}
    t0 = time.time()
    for entry in src:
        vid = entry.get("video_id", entry.get("id"))
        if vid in done and not a.force:
            continue
        totals["clips"] += 1
        dense_entry, stats = densify_manifest_entry(
            entry, fps_target=a.fps_target, pad_sec=a.pad_sec, detector=detector,
        )
        added = stats.get("added_samples", 0)
        if added > 0:
            totals["densified"] += 1
            totals["added"] += added
        print(f"  {vid[:8]}  +{added} samples ({stats.get('densified_persons', 0)} persons)"
              f"{' [no-op]' if added == 0 else ''}")
        results.append(dense_entry)
        # atomic, resumable checkpoint
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results))
        tmp.replace(out_path)

    print(f"\n[densify] {totals['clips']} processed, {totals['densified']} densified, "
          f"+{totals['added']} samples in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
