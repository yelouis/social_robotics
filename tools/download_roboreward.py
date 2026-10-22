from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional

from huggingface_hub import HfApi, snapshot_download

from config import DATA_ROOT
from shared.fsutil import data_glob

# Disable experimental xet backend which causes excessive per-file API calls and 429 rate limits
os.environ["HF_HUB_DISABLE_XET"] = "1"


REPO_ID = "teetone/RoboReward"
REVISION = "469b9af76e8539e8d2ac553b081307738fd92ca5"
EXPECTED_FILES = 54140
EXPECTED_BYTES = 43204474744
EXPECTED_MP4 = 54135
REQUIRED_EXTRA_BYTES = 50 * (1024**3)
TOTAL_REQUIRED_BYTES = EXPECTED_BYTES + REQUIRED_EXTRA_BYTES


def check_disk_space(target_dir: Path, total_required_bytes: int = TOTAL_REQUIRED_BYTES) -> None:
    """Verify free space >= dataset bytes + 50 GiB."""
    usage = shutil.disk_usage(target_dir)
    free_bytes = usage.free
    free_gib = free_bytes / (1024**3)
    req_gib = total_required_bytes / (1024**3)
    if free_bytes < total_required_bytes:
        raise RuntimeError(
            f"Disk rule violated: free space {free_gib:.1f} GiB < {req_gib:.1f} GiB "
            f"({total_required_bytes} bytes needed: {EXPECTED_BYTES} dataset + 50 GiB buffer)"
        )
    print(f"Disk check OK: {free_gib:.1f} GiB free (required: {req_gib:.1f} GiB)")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_download(target_dir: Path, api: Optional[HfApi] = None) -> Dict[str, Any]:
    api = api or HfApi()
    import time
    info = None
    for attempt in range(5):
        try:
            info = api.dataset_info(repo_id=REPO_ID, revision=REVISION, files_metadata=True)
            break
        except Exception as e:
            if attempt == 4:
                raise
            print(f"dataset_info attempt {attempt+1}/5 error: {e}. Sleeping 60s...")
            time.sleep(60)

    assert info is not None
    expected_siblings = {s.rfilename: s.size for s in info.siblings if s.size is not None}

    # Verify each expected sibling exists with exact size
    for rfilename, exp_size in expected_siblings.items():
        local_path = target_dir / rfilename
        if not local_path.is_file():
            raise FileNotFoundError(f"Missing file from dataset: {rfilename}")
        act_size = local_path.stat().st_size
        if act_size != exp_size:
            raise ValueError(f"File size mismatch for {rfilename}: {act_size} != {exp_size}")

    # Count all non-AppleDouble regular files belonging to the dataset (excluding cache and tracking files)
    all_files = [
        p for p in data_glob(target_dir, "*", recursive=True)
        if p.is_file() and ".cache" not in p.parts and p.name not in ("DOWNLOAD.json", "progress.json")
    ]
    n_files = len(all_files)
    total_bytes = sum(p.stat().st_size for p in all_files)
    n_mp4 = sum(1 for p in all_files if p.suffix == ".mp4")

    print(f"Verification counts in {target_dir}:")
    print(f"  Files: {n_files} (expected {len(expected_siblings)})")
    print(f"  Bytes: {total_bytes} (expected {sum(expected_siblings.values())})")
    print(f"  .mp4 : {n_mp4}")

    if n_files != len(expected_siblings):
        raise ValueError(f"File count mismatch: {n_files} != {len(expected_siblings)}")
    if total_bytes != sum(expected_siblings.values()):
        raise ValueError(f"Byte count mismatch: {total_bytes} != {sum(expected_siblings.values())}")

    # Compute sha256 for metadata files
    metadata_sha256 = {}
    for split in ["train", "val", "test"]:
        split_meta = target_dir / split / "metadata.jsonl"
        if split_meta.is_file():
            metadata_sha256[split] = file_sha256(split_meta)

    usage = shutil.disk_usage(target_dir)
    free_gib_after = round(usage.free / (1024**3), 2)

    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "n_files": n_files,
        "bytes": total_bytes,
        "n_mp4": n_mp4,
        "metadata_sha256": metadata_sha256,
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "free_gib_after": free_gib_after,
    }


def update_progress(target_dir: Path, progress_path: Optional[Path] = None) -> None:
    p_path = progress_path or (target_dir / "progress.json")
    all_files = [
        str(p.relative_to(target_dir))
        for p in data_glob(target_dir, "*", recursive=True)
        if p.is_file() and p.name not in ("DOWNLOAD.json", "progress.json")
    ]
    p_path.parent.mkdir(parents=True, exist_ok=True)
    p_path.write_text(json.dumps(all_files), encoding="utf-8")


def download_roboreward(
    target_dir: Optional[Path] = None,
    verify_only: bool = False,
    api: Optional[HfApi] = None,
    progress_file: Optional[Path] = None,
) -> Path:
    dest = target_dir or (DATA_ROOT / "raw" / "roboreward")
    dest.mkdir(parents=True, exist_ok=True)

    check_disk_space(dest)

    if not verify_only:
        print(f"Downloading {REPO_ID} (rev: {REVISION}) into {dest}...")
        import time
        max_attempts = 15
        for attempt in range(1, max_attempts + 1):
            try:
                snapshot_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=str(dest),
                    max_workers=4,
                    token=True,
                )
                all_files = [
                    p for p in data_glob(dest, "*", recursive=True)
                    if p.is_file() and p.name not in ("DOWNLOAD.json", "progress.json")
                ]
                if target_dir is None and len(all_files) < EXPECTED_FILES:
                    print(f"snapshot_download returned prematurely: {len(all_files)}/{EXPECTED_FILES} files. Retrying...")
                    raise RuntimeError(f"Download incomplete: {len(all_files)}/{EXPECTED_FILES} files")
                break
            except Exception as e:
                update_progress(dest, progress_file)
                if attempt == max_attempts:
                    print(f"Download failed after {max_attempts} attempts: {e}")
                    raise
                sleep_s = min(300, 30 * attempt)
                print(f"Attempt {attempt}/{max_attempts} error: {e}. Sleeping {sleep_s}s...")
                time.sleep(sleep_s)

        update_progress(dest, progress_file)

    meta = verify_download(dest, api=api)
    download_json = dest / "DOWNLOAD.json"
    download_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {download_json}")
    print(f"roboreward: {meta['n_files']} files, {meta['bytes']} bytes, {meta['n_mp4']} mp4s")

    return download_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify RoboReward dataset for Wave D")
    parser.add_argument("--target-dir", default=None, type=Path, help="Target download directory")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files on disk")
    parser.add_argument("--progress-file", default=None, type=Path, help="Path to progress.json")
    args = parser.parse_args()

    download_roboreward(
        target_dir=args.target_dir,
        verify_only=args.verify_only,
        progress_file=args.progress_file,
    )


if __name__ == "__main__":
    main()
