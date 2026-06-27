#!/usr/bin/env python
"""One shard worker for the parallel layer runner (tools/run_parallel_layer.py).

Runs a heavy per-bystander Layer-03 pipeline on a *subset* (shard) of the
manifest's clips, writing to a per-shard output file. Run as an isolated
subprocess so it gets its OWN MPS context — a crash here cannot take down the
siblings or the orchestrator, and there is no fork-after-MPS-init hazard.

Resumable: the pipeline's own run() skips clips already in the shard output.
"""
import sys
import time
import argparse
import importlib

sys.path.insert(0, "/Users/louisye/Desktop/Louis/social_robotics/src")

# layer id -> (module, pipeline class). Both layers take (manifest, output, force).
PIPELINES = {
    "03d": ("layer_03d_proxemic_kinematics.pipeline", "ProxemicKinematicsPipeline"),
    "03f": ("layer_03f_motor_resonance.pipeline", "MotorResonancePipeline"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, choices=list(PIPELINES))
    ap.add_argument("--manifest", required=True, help="shard manifest (a clip subset)")
    ap.add_argument("--output", required=True, help="per-shard output json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    mod, cls = PIPELINES[a.layer]
    Pipe = getattr(importlib.import_module(mod), cls)

    t0 = time.time()
    print(f"[shard:{a.output}] init {a.layer}", flush=True)
    Pipe(a.manifest, a.output, force=a.force).run()
    print(f"[shard:{a.output}] DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
