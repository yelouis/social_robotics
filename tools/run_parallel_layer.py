#!/usr/bin/env python
"""Parallel runner for the heavy per-bystander Layer-03 pipelines (03d, 03f).

WHY: these layers alternate, in a single thread, between slow random-seek video
decode (CPU) and Depth/SAM/YOLO inference (MPS GPU). Neither saturates — on the
M4 Max the run sits at ~1.3 / 16 cores and ~7% RAM. Sharding the clips across N
isolated worker processes lets their decode overlap each other's GPU inference,
filling the single GPU's idle gaps. Gains saturate around N=3-4 (one physical GPU).

SAFETY (the reason this is a harness and not just `&`-ing N runs):
  * isolated subprocess workers — own MPS context, crash-isolated;
  * STAGGERED startup (--stagger-sec) so N x ~4 GB model loads don't spike at once;
  * a pre-launch RAM floor (--min-free-ram) — the orchestrator will not start the
    next worker while free RAM is below it;
  * bounded N; weighted sharding (balance by bystander count); resumable.

Use --only-clips / --max-clips to STRESS-TEST on the heaviest clips first (verify
memory stays safe + nothing crashes) before committing to the full run.

Example (heavy-load test, 3 workers, the explosion clips):
  python tools/run_parallel_layer.py --layer 03f \
      --manifest "$RUN/03a/input_top200.json" \
      --output   "$RUN/03a/03f_motor_resonance_result.json" \
      --workers 3 --only-clips 46bcee63-...,1fe55d7f-... --min-free-ram 25
"""
import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "_shard_worker.py")
PYTHON = "/Users/louisye/Desktop/Louis/social_robotics/venv/bin/python"
SSD_HF_CACHE = "/Volumes/Extreme SSD/huggingface_cache"


def free_ram_pct():
    """System-wide free RAM %. Falls back to 100 (don't block) if unreadable."""
    try:
        out = subprocess.check_output(["memory_pressure"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "free percentage" in line.lower():
                digits = "".join(c for c in line.split(":")[-1] if c.isdigit())
                return int(digits) if digits else 100
    except Exception:
        pass
    return 100


def _weight(entry):
    """Cost proxy: count of positive-id (genuine, kept) bystander tracks."""
    return sum(1 for b in entry.get("bystander_detections", [])
               if (b.get("person_id") if b.get("person_id") is not None else -1) >= 0) or 1


def shard_clips(manifest, n, only=None, max_clips=None):
    """Greedy weighted bin-pack: heaviest clips first onto the least-loaded shard.
    Deterministic, so a re-run produces the same shards (resume-safe)."""
    clips = [e for e in manifest if not e.get("synthetic")]
    if only:
        clips = [e for e in clips if e.get("video_id") in only]
    clips.sort(key=_weight, reverse=True)
    if max_clips:
        clips = clips[:max_clips]
    shards = [[] for _ in range(n)]
    load = [0] * n
    for e in clips:
        i = min(range(n), key=lambda k: load[k])
        shards[i].append(e)
        load[i] += _weight(e)
    return shards, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, choices=["03d", "03f"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stagger-sec", type=int, default=30,
                    help="delay between worker launches (avoid simultaneous model-load memory spike)")
    ap.add_argument("--min-free-ram", type=int, default=20,
                    help="hold the next worker launch while free RAM %% is below this")
    ap.add_argument("--only-clips", help="comma-separated video_ids (heavy-load stress test)")
    ap.add_argument("--max-clips", type=int, help="cap total clips processed (stress test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.environ["HF_HOME"] = SSD_HF_CACHE  # inherited by workers

    manifest = json.load(open(a.manifest))
    only = set(a.only_clips.split(",")) if a.only_clips else None
    shards, load = shard_clips(manifest, a.workers, only, a.max_clips)
    workdir = a.output + ".parallel"
    os.makedirs(workdir, exist_ok=True)

    print(f"[orch] layer={a.layer} workers={a.workers} | shard weights={load} | "
          f"clips/shard={[len(s) for s in shards]}", flush=True)

    procs, shard_outs = [], []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        sm = os.path.join(workdir, f"shard{i}.manifest.json")
        so = os.path.join(workdir, f"shard{i}.result.json")
        json.dump(shard, open(sm, "w"))
        shard_outs.append(so)

        # SAFETY: wait for RAM headroom before adding another ~4 GB worker.
        waited = 0
        while free_ram_pct() < a.min_free_ram and waited < 1800:
            print(f"[orch] free RAM {free_ram_pct()}% < {a.min_free_ram}% — holding shard {i}…", flush=True)
            time.sleep(30)
            waited += 30

        cmd = [PYTHON, "-u", WORKER, "--layer", a.layer, "--manifest", sm, "--output", so]
        if a.force:
            cmd.append("--force")
        log = open(os.path.join(workdir, f"shard{i}.log"), "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        procs.append((i, p))
        print(f"[orch] launched shard {i}: {len(shard)} clips, pid {p.pid}, free RAM {free_ram_pct()}%", flush=True)
        if i < len(shards) - 1:
            time.sleep(a.stagger_sec)  # stagger model loads

    for i, p in procs:
        p.wait()
        print(f"[orch] shard {i} exited {p.returncode}", flush=True)

    # Merge per-shard outputs into the final result.
    merged = []
    for so in shard_outs:
        if os.path.exists(so):
            try:
                merged.extend(json.load(open(so)))
            except Exception as e:
                print(f"[orch] WARN: could not read {so}: {e}", flush=True)
    tmp = a.output + ".tmp"
    json.dump(merged, open(tmp, "w"), indent=2)
    os.replace(tmp, a.output)
    print(f"[orch] merged {len(merged)} clips -> {a.output}", flush=True)


if __name__ == "__main__":
    main()
