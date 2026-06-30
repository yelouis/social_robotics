#!/usr/bin/env python3
"""
Corpus-landing runner (top-200 scope) for the window-dense bystander change.

  1. Densify the top-200 run manifest (src/shared/dense_detect.py), serial.
  2. Re-run 03f Motor Resonance on the dense manifest via the PARALLEL runner
     (tools/run_parallel_layer.py, workers=3 -> ~3x; no 03f code change).
  3. Re-run 03d Proxemic Kinematics on the dense manifest via the parallel
     runner (now emits the additive proxemic_trajectory_* fields).
  4. Compare new vs the existing (sparse) top-200 results; write a report.

Stops BEFORE any Hugging Face re-publish (outward-facing -> needs confirmation).
Resumable: densify skips an existing dense manifest; the parallel runner resumes
per-clip. Multi-hour (03d depth+SAM x 200) -> launch under a detached daemon.
"""
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PYTHON = str(ROOT / "venv/bin/python")

from src.shared.dense_detect import densify_manifest_entry, _DenseDetector

SSD = Path("/Volumes/Extreme SSD/social_robotics/full_run_2026_06_18/03a")
BASE = SSD / "input_top200.json"
OLD_03F = SSD / "03f_motor_resonance_result.json"
OLD_03D = SSD / "03d_proxemic_kinematics_result.json"

RUN = ROOT / "e2e_reports/2026_06_29_density_landing_top200"
RUN.mkdir(parents=True, exist_ok=True)
DENSE = RUN / "input_top200.dense.json"
NEW_03F = RUN / "03f_motor_resonance_result.json"
NEW_03D = RUN / "03d_proxemic_kinematics_result.json"


def stage1_densify():
    if DENSE.exists():
        print(f"[1/4] dense manifest exists ({DENSE.name}); skipping densify", flush=True)
        return
    base = json.loads(BASE.read_text())
    det = _DenseDetector()
    out, added = [], 0
    t0 = time.time()
    for i, e in enumerate(base, 1):
        e.setdefault("video_id", e.get("id"))
        de, st = densify_manifest_entry(e, detector=det)
        added += st.get("added_samples", 0)
        out.append(de)
        if i % 25 == 0 or i == len(base):
            print(f"[1/4] densify {i}/{len(base)}  (+{added} samples, {time.time()-t0:.0f}s)", flush=True)
    tmp = DENSE.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(DENSE)
    print(f"[1/4] densified {len(out)} clips (+{added} samples) in {time.time()-t0:.0f}s", flush=True)


def stage_parallel(layer, out_path):
    print(f"[run] {layer} (parallel, workers=3) on dense manifest -> {out_path.name}", flush=True)
    t0 = time.time()
    subprocess.run(
        [PYTHON, str(ROOT / "tools/run_parallel_layer.py"),
         "--layer", layer, "--manifest", str(DENSE), "--output", str(out_path), "--workers", "3"],
        check=True, cwd=str(ROOT),
    )
    print(f"[run] {layer} done in {time.time()-t0:.0f}s", flush=True)


def _by_id(path):
    p = Path(path)
    return {r["video_id"]: r for r in json.loads(p.read_text())} if p.exists() else {}


def stage4_compare():
    lines = ["", "==========  DENSITY LANDING — top-200 (new dense vs existing sparse)  =========="]

    old, new = _by_id(OLD_03F), _by_id(NEW_03F)
    def f_stats(res):
        clips = sum(1 for r in res.values() if r.get("tasks_analyzed"))
        reson = sum(p.get("motor_resonance_detected", False)
                    for r in res.values() for t in r.get("tasks_analyzed", []) for p in t.get("per_person", []))
        return clips, reson
    oc, orr = f_stats(old); nc, nr = f_stats(new)
    lines += ["", "03f Motor Resonance:",
              f"  clips with data          {oc:>4}  ->  {nc:>4}",
              f"  motor_resonance_detected {orr:>4}  ->  {nr:>4}"]

    old, new = _by_id(OLD_03D), _by_id(NEW_03D)
    def d_stats(res):
        clips = sum(1 for r in res.values() if r.get("tasks_analyzed"))
        nonneutral = sum(1 for r in res.values() for t in r.get("tasks_analyzed", [])
                         for p in t.get("per_person", []) if p.get("classified_action") not in (None, "Neutral"))
        shapes = Counter(p.get("proxemic_trajectory_shape")
                         for r in res.values() for t in r.get("tasks_analyzed", []) for p in t.get("per_person", [])
                         if p.get("proxemic_trajectory_shape"))
        return clips, nonneutral, shapes
    oc, onn, _ = d_stats(old); nc, nn, shapes = d_stats(new)
    lines += ["", "03d Proxemic Kinematics:",
              f"  clips with data          {oc:>4}  ->  {nc:>4}",
              f"  non-Neutral actions      {onn:>4}  ->  {nn:>4}",
              f"  NEW proxemic_trajectory_shape distribution: {dict(shapes)}"]

    report = "\n".join(lines)
    print(report, flush=True)
    (RUN / "LANDING_REPORT.txt").write_text(report + "\n")


if __name__ == "__main__":
    stage1_densify()
    stage_parallel("03f", NEW_03F)
    stage_parallel("03d", NEW_03D)
    stage4_compare()
    print("\n[run_density_landing] LANDING DONE", flush=True)
