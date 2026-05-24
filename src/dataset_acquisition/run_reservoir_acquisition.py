"""Node 01+02 reservoir acquisition (Ego4D; Layer 1a disabled).

Download Ego4D in batches, run the FULL Node 02 social-presence filter to
*score* each video (`social_presence_score`), and retain only the top-K
(`ScoreReservoir`) within a hard disk budget. Non-passers (no social presence
or no task labels) and reservoir-evicted clips are purged, so the working set
stays bounded to the highest-social-presence videos.

Save points make it resumable and incremental:
  - `processed_uids.json` — every UID seen (pass or fail); re-runs never
    re-download or re-process these.
  - `reservoir_state.json` — the retained set (uid -> score -> path).
  - `filtered_manifest.json` — the retained set in full Layer 03 schema (the
    03 save-state), refreshed as the reservoir changes.

Usage:
  # full run: process 2000 UIDs (500 local re-scored + ~1500 new), keep top-1000
  python -m dataset_acquisition.run_reservoir_acquisition --target 2000 --cap 1000
  # isolated dry-run smoke on local videos (no download, no deletion, temp state):
  python -m dataset_acquisition.run_reservoir_acquisition --smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Layer 1a is disabled for corpus acquisition (Ego4D only). Set before imports
# that read it. Keep the model-tier banner so the active VLM is visible.
os.environ.setdefault("SAF_RUN_SYNTHETIC_QA", "0")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from dataset_acquisition.downloader import Ego4DDownloader  # noqa: E402
from dataset_acquisition.reservoir import ScoreReservoir, DEFAULT_CAP, DEFAULT_DISK_BUDGET_BYTES  # noqa: E402
from filtering_and_labeling.pipeline import FilteringPipeline  # noqa: E402

SSD_ROOT = config.OUTPUT_DIR.parent  # /Volumes/Extreme SSD/social_robotics
PROD_PROCESSED = config.OUTPUT_DIR / "ego4d" / "processed_uids.json"
PROD_STATE = SSD_ROOT / "reservoir_state.json"
PROD_MANIFEST = SSD_ROOT / "filtered_manifest.json"


def _load_set(path: Path) -> set:
    if path.exists():
        try:
            return set(json.load(open(path)))
        except Exception:
            return set()
    return set()


def _save_set(path: Path, s: set):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    json.dump(sorted(s), open(tmp, "w"))
    os.replace(tmp, path)


def find_local_path(output_path: Path, uid: str):
    for d in (output_path / "v2" / "full_scale", output_path):
        p = d / f"{uid}.mp4"
        if p.exists() and not p.name.startswith("._"):
            return p
    for p in output_path.rglob(f"{uid}.mp4"):
        if not p.name.startswith("._"):
            return p
    return None


def list_local_uids(output_path: Path):
    uids = []
    for p in sorted(output_path.rglob("*.mp4")):
        if not p.name.startswith("._"):
            uids.append(p.stem)
    return uids


def score_video(pipe: FilteringPipeline, path: Path, uid: str):
    """Run the full Node 02 path. Returns (passed, score, manifest_record)."""
    bystanders, hands = pipe.social_presence_filter(path)
    if not bystanders:
        return (False, 0.0, None)
    entry = {
        "id": uid, "video_id": uid, "dataset": "ego4d", "file_path": str(path),
        "bystander_detections": bystanders, "hand_detections": hands,
    }
    rec = pipe.process_video_vlm_pass(entry)  # adds identified_tasks + score; None if no tasks
    if rec is None:
        return (False, 0.0, None)
    return (True, float(rec.get("social_presence_score", 0.0)), rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000, help="UIDs to process this run (incl. local).")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--disk-budget-gib", type=float, default=DEFAULT_DISK_BUDGET_BYTES / 2**30)
    ap.add_argument("--batch", type=int, default=25, help="Download batch size.")
    ap.add_argument("--no-download", action="store_true", help="Only score already-local UIDs.")
    ap.add_argument("--dry-run", action="store_true", help="Never delete files (log evictions/purges only).")
    ap.add_argument("--limit", type=int, default=None, help="Hard cap on UIDs processed (smoke).")
    ap.add_argument("--smoke", action="store_true",
                    help="Isolated dry-run on local videos: --no-download --dry-run, temp state, small cap/limit.")
    args = ap.parse_args()

    if args.smoke:
        args.no_download = True
        args.dry_run = True
        if args.limit is None:
            args.limit = 8
        if args.cap == DEFAULT_CAP:
            args.cap = 3
        processed_path = ROOT / "scratch" / "reservoir_smoke_processed.json"
        state_path = ROOT / "scratch" / "reservoir_smoke_state.json"
        manifest_path = ROOT / "scratch" / "reservoir_smoke_manifest.json"
        print("[reservoir] SMOKE: isolated temp state, dry-run, no download.")
    else:
        processed_path, state_path, manifest_path = PROD_PROCESSED, PROD_STATE, PROD_MANIFEST

    dl = Ego4DDownloader()  # download only; we run our own full filter + scoring + reservoir
    dl.filter_on_the_fly = False  # skip the StreamingFilter purge inside download()
    dl.filterer = None
    pipe = FilteringPipeline(input_manifest_path=str(manifest_path),
                             output_manifest_path=str(ROOT / "scratch" / "_reservoir_pipe_unused.json"),
                             force=True)
    reservoir = ScoreReservoir(cap=args.cap, disk_budget_bytes=int(args.disk_budget_gib * 2**30),
                               state_path=state_path, dry_run=args.dry_run).load()
    processed = _load_set(processed_path)
    print(f"[reservoir] cap={args.cap} disk_budget={args.disk_budget_gib:.0f}GiB "
          f"dry_run={args.dry_run} | resuming: {len(reservoir.entries)} kept, {len(processed)} processed")

    # Build the work list: local-already-downloaded first (no download cost),
    # then new corpus UIDs (download), excluding anything already processed.
    local_uids = [u for u in list_local_uids(dl.output_path) if u not in processed]
    if args.no_download:
        work = local_uids
    else:
        local_set = set(local_uids)
        new_uids = [u for u in dl.get_all_uids() if u not in processed and u not in local_set]
        remaining = max(0, args.target - len(processed))
        work = (local_uids + new_uids)[:remaining]
    if args.limit is not None:
        work = work[: args.limit]
    print(f"[reservoir] work list: {len(work)} UIDs ({len(local_uids)} local available)")

    def purge(path):
        if args.dry_run or path is None:
            return
        try:
            Path(path).unlink()
        except OSError:
            pass

    kept = rejected = dropped = skipped_solo = missing = 0
    start = time.time()
    for idx, uid in enumerate(work, 1):
        if uid in processed:
            continue
        if pipe._is_likely_solo_by_metadata(uid):
            skipped_solo += 1
            purge(find_local_path(dl.output_path, uid))
            processed.add(uid)
            print(f"[{idx}/{len(work)}] SKIP solo {uid}", flush=True)
            _save_set(processed_path, processed)
            continue
        path = find_local_path(dl.output_path, uid)
        if path is None and not args.no_download:
            try:
                dl.download(video_uids=[uid])
            except Exception as e:
                print(f"[{idx}/{len(work)}] download error {uid}: {e}", flush=True)
            path = find_local_path(dl.output_path, uid)
        if path is None:
            missing += 1
            processed.add(uid)
            _save_set(processed_path, processed)
            print(f"[{idx}/{len(work)}] MISSING {uid}", flush=True)
            continue

        t0 = time.time()
        try:
            passed, score, rec = score_video(pipe, path, uid)
        except Exception as e:
            print(f"[{idx}/{len(work)}] ERROR {uid}: {e}", flush=True)
            processed.add(uid)
            _save_set(processed_path, processed)
            continue
        dt = time.time() - t0

        if not passed:
            dropped += 1
            purge(path)
            print(f"[{idx}/{len(work)}] DROP {uid} (no social presence / no task) t={dt:.0f}s", flush=True)
        else:
            size = path.stat().st_size if path.exists() else 0
            status, displaced = reservoir.consider(uid, score, path, size, rec)
            if status == "kept":
                kept += 1
                msg = f"KEEP {uid} score={score:.2f}"
                if displaced:
                    msg += f" (evicted {len(displaced)}: min now {reservoir.min_score():.2f})"
            else:
                rejected += 1
                msg = f"REJECT {uid} score={score:.2f} (< reservoir min {reservoir.min_score():.2f})"
            print(f"[{idx}/{len(work)}] {msg} t={dt:.0f}s | kept={len(reservoir.entries)} "
                  f"disk={reservoir.total_bytes()/2**30:.1f}GiB", flush=True)

        processed.add(uid)
        if idx % 5 == 0 or idx == len(work):
            _save_set(processed_path, processed)
            reservoir.save()
            reservoir.export_manifest(manifest_path)

    _save_set(processed_path, processed)
    reservoir.save()
    n_manifest = reservoir.export_manifest(manifest_path)
    mins = (time.time() - start) / 60
    print(f"\n[reservoir] Done in {mins:.1f} min. processed_this_run={len([u for u in work])} "
          f"kept={kept} rejected={rejected} dropped={dropped} solo={skipped_solo} missing={missing}")
    print(f"[reservoir] reservoir now holds {len(reservoir.entries)} videos "
          f"({reservoir.total_bytes()/2**30:.1f} GiB); manifest={n_manifest} entries -> {manifest_path}")


if __name__ == "__main__":
    main()
