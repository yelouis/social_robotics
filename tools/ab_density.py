#!/usr/bin/env python3
"""
A/B test: does densifying Node-02 bystander boxes (inside the reaction windows)
change the kinematic Layer-03 outputs?

Sparse arm = original manifest boxes (~1 every 3-12s).
Dense arm  = same boxes, re-detected densely (YOLO+ByteTrack) inside each reaction
             window and matched back to the manifest person_ids (src/shared/dense_detect.py).

Densification is layer-agnostic, so we build the sparse/dense manifests ONCE and
run every requested layer on both. Each layer's pipeline is resumable (force=False),
so a restart of this harness continues where it left off.

Usage:
    python tools/ab_density.py --out DIR --n 50 --layers 03f,03d
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer_05_visualizer.catalog import build_catalog
from src.shared.dense_detect import densify_manifest_entry, _DenseDetector

LAYERS = {
    "03f": {
        "import": ("src.layer_03f_motor_resonance.pipeline", "MotorResonancePipeline"),
        "finding": lambda p: bool(p.get("motor_resonance_detected")),
        "scalar": lambda p: round(p.get("bystander_pose_velocity_peak", 0), 2),
        "trusted": lambda p: p.get("bystander_pose_velocity_peak", 0) > 0,
        "finding_label": "motor_resonance_detected",
        "trusted_label": "rows with trusted velocity (>0)",
    },
    "03d": {
        "import": ("src.layer_03d_proxemic_kinematics.pipeline", "ProxemicKinematicsPipeline"),
        "finding": lambda p: p.get("classified_action") not in (None, "Neutral"),
        "scalar": lambda p: p.get("classified_action", "-"),
        "trusted": lambda p: p.get("proxemic_confidence", 0) > 0,
        "finding_label": "non-Neutral classified_action (Approach/Avoidance)",
        "trusted_label": "rows with proxemic_confidence > 0",
    },
}


def _load_pipeline(layer):
    mod, cls = LAYERS[layer]["import"]
    import importlib
    return getattr(importlib.import_module(mod), cls)


def build_manifests(out: Path, n: int):
    sparse_p, dense_p = out / "sparse_manifest.json", out / "dense_manifest.json"
    if sparse_p.exists() and dense_p.exists():
        print(f"[manifests] reusing existing {sparse_p.name} / {dense_p.name}")
        return sparse_p, dense_p

    cat = build_catalog([Path("e2e_reports")])
    entries = [e.manifest_entry for e in cat.values()
               if e.video_path and Path(e.video_path).exists()
               and e.manifest_entry.get("bystander_detections")
               and any(t.get("task_temporal_metadata", {}).get("task_reaction_window_sec")
                       for t in e.manifest_entry.get("identified_tasks", []))]
    for e in entries:
        e.setdefault("video_id", e.get("id"))
    print(f"[manifests] {len(entries)} candidate clips; densifying until {n} differ...")

    detector = _DenseDetector()
    sparse_set, dense_set = [], []
    t0 = time.time()
    for e in entries:
        if len(dense_set) >= n:
            break
        dense_e, stats = densify_manifest_entry(e, detector=detector)
        if stats.get("added_samples", 0) > 0:
            sparse_set.append(e)
            dense_set.append(dense_e)
            print(f"  + {e['video_id'][:8]} densified {stats['densified_persons']} person(s), "
                  f"+{stats['added_samples']} samples  [{len(dense_set)}/{n}]")
    print(f"[manifests] densification done in {time.time()-t0:.0f}s; A/B set = {len(dense_set)}")
    sparse_p.write_text(json.dumps(sparse_set))
    dense_p.write_text(json.dumps(dense_set))
    return sparse_p, dense_p


def run_layer(layer, manifest, out: Path, tag):
    Pipeline = _load_pipeline(layer)
    res_path = out / f"{layer}_{tag}.json"
    print(f"--- {layer} [{tag}] -> {res_path.name} ---", flush=True)
    t = time.time()
    Pipeline(manifest, res_path, force=False).run()   # resumable
    print(f"    {layer} [{tag}] done in {time.time()-t:.0f}s", flush=True)
    return {r["video_id"]: r for r in json.loads(res_path.read_text())}


def summarize(res, spec):
    clips, rows, trusted, findings = 0, 0, 0, 0
    for r in res.values():
        ta = r.get("tasks_analyzed", [])
        if ta:
            clips += 1
        for t in ta:
            for p in t.get("per_person", []):
                rows += 1
                if spec["trusted"](p):
                    trusted += 1
                if spec["finding"](p):
                    findings += 1
    return clips, rows, trusted, findings


def report(layer, sparse_res, dense_res, out: Path):
    spec = LAYERS[layer]
    s, d = summarize(sparse_res, spec), summarize(dense_res, spec)
    lines = []
    lines.append(f"\n================  {layer} A/B RESULTS (N={len(sparse_res)})  ================")
    hdr = f"{'metric':<46}{'SPARSE':>8}{'DENSE':>8}"
    lines.append(hdr); lines.append("-" * len(hdr))
    for lab, sv, dv in zip(
        ["clips with data (not skipped)", "per-person rows total",
         spec["trusted_label"], spec["finding_label"]], s, d):
        lines.append(f"{lab:<46}{sv:>8}{dv:>8}")
    lines.append("\nper-clip scalar (sparse -> dense):")
    for vid in sorted(sparse_res):
        def vals(res):
            o = [spec["scalar"](p) for t in res.get(vid, {}).get("tasks_analyzed", [])
                 for p in t.get("per_person", [])]
            return o or [res.get(vid, {}).get("skipped_reason", "-")]
        sv, dv = vals(sparse_res), vals(dense_res)
        lines.append(f"  {vid[:8]}  {str(sv):<30} -> {dv}{'  *' if sv != dv else ''}")
    text = "\n".join(lines)
    print(text, flush=True)
    (out / f"{layer}_REPORT.txt").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--layers", default="03f,03d")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sparse_m, dense_m = build_manifests(out, a.n)
    for layer in a.layers.split(","):
        layer = layer.strip()
        if layer not in LAYERS:
            print(f"!! unknown layer {layer}, skipping"); continue
        sparse_res = run_layer(layer, sparse_m, out, "sparse")
        dense_res = run_layer(layer, dense_m, out, "dense")
        report(layer, sparse_res, dense_res, out)
    print("\n[ab_density] ALL DONE")


if __name__ == "__main__":
    main()
