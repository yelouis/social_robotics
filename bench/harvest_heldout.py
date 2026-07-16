"""SRB v0 — download the held-out Ego4D clips and run the Node-02 engine
pre-pass (moment proposer, docs/07 §0 role 1).

Chain per batch: ego4d CLI download (via the engine's Ego4DDownloader, with
its social-presence purge gate ON so dead clips never hit disk quota) →
then, once all batches land, build a registry of the survivors and run the
full Node-02 SocialFilterPipeline into `bench_v0/heldout_manifest.json`
(Layer-03 schema — the input 02b/03x expect).

This file intentionally shells INTO the engine's acquisition/filtering
drivers rather than reimplementing them; the docs/07 §0 import boundary
applies to benchmark *item* code, and this is the engine acting in its
sanctioned moment-proposer role, orchestrated from bench/.

Usage:
    python bench/harvest_heldout.py [--batch-size 5] [--skip-download]
Resumable: already-downloaded uids are skipped; Node-02 resumes by video_id.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from srb_common import BENCH_DATA, SSD_ROOT, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))


def find_video(uid: str):
    for pat in [SSD_ROOT / "raw_videos" / "ego4d" / "v2" / "full_scale" / f"{uid}.mp4",
                SSD_ROOT / "raw_videos" / "ego4d" / "full_scale" / f"{uid}.mp4"]:
        if pat.exists():
            return pat
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--skip-download", action="store_true",
                    help="Only (re)run Node-02 on already-downloaded clips.")
    a = ap.parse_args()

    with open(BENCH_DATA / "heldout_download_uids.json") as f:
        picked = json.load(f)
    uids = [p["video_uid"] for p in picked]

    if not a.skip_download:
        from dataset_acquisition.downloader import Ego4DDownloader
        dl = Ego4DDownloader()
        dl.filter_on_the_fly = True          # purge no-social clips at the door
        todo = [u for u in uids if find_video(u) is None]
        print(f"[harvest] {len(uids)} selected; {len(uids)-len(todo)} already local; downloading {len(todo)}")
        for i in range(0, len(todo), a.batch_size):
            batch = todo[i:i + a.batch_size]
            print(f"[harvest] batch {i//a.batch_size + 1}: {batch}")
            t0 = time.time()
            dl.download(video_uids=batch)
            print(f"[harvest] batch done in {time.time()-t0:.0f}s")

    survivors = [(u, find_video(u)) for u in uids]
    survivors = [(u, p) for u, p in survivors if p is not None]
    print(f"[harvest] survivors after social-presence purge: {len(survivors)}/{len(uids)}")

    registry = [{"id": u, "video_id": u, "dataset": "ego4d",
                 "file_path": str(p), "video_path": str(p),
                 "file_size": p.stat().st_size} for u, p in survivors]
    reg_path = BENCH_DATA / "heldout_registry.json"
    reg_path.write_text(json.dumps(registry, indent=2))

    # Node-02 full pass -> Layer-03 schema manifest (resumable).
    from filtering_and_labeling.pipeline import FilteringPipeline
    out = BENCH_DATA / "heldout_manifest.json"
    print(f"[harvest] Node-02 over {len(registry)} clips -> {out}")
    FilteringPipeline(reg_path, out).run()
    with open(out) as f:
        kept = json.load(f)
    print(f"[harvest] Node-02 kept {len(kept)} clips with tasks+bystanders")


if __name__ == "__main__":
    main()
