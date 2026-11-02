from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import zipfile

from config import DATA_ROOT


ZENODO_RECORD = "14590226"
DOI = "10.5281/zenodo.14590226"
LICENSE = "CC-BY-4.0"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/Dataset.zip/content"

EXPECTED_ZIP_BYTES = 37219808401
EXPECTED_ZIP_MD5 = "334caac04f3e607ac91182dea7a03929"
N_PARTICIPANTS = 55

STRATEGY_FOLDERS = (
    "C1-Fixed-Low",
    "C2-Fixed-Medium",
    "C3-Fixed-High",
    "D1-Decay-Smooth",
    "D2-Decay-Rapid",
)

REQUIRED_FILES = (
    "analysis.csv",
    "body.csv",
    "facetorch.csv",
    "gaze.csv",
    "hume.csv",
    "openface.csv",
    "questionnaire.csv",
    "speech.csv",
    "time.csv",
    "video_cam1.mp4",
)


def check_disk_space(target_dir: Path, min_free_bytes: int = 100 * (1024**3)) -> None:
    """Verify target drive has sufficient free space."""
    usage = shutil.disk_usage(target_dir)
    free_gib = usage.free / (1024**3)
    min_gib = min_free_bytes / (1024**3)
    if usage.free < min_free_bytes:
        raise RuntimeError(
            f"Insufficient disk space in {target_dir}: {free_gib:.1f} GiB free < {min_gib:.1f} GiB required"
        )
    print(f"Disk space OK: {free_gib:.1f} GiB free in {target_dir} (required: {min_gib:.1f} GiB)")


def file_md5(path: Path) -> str:
    """Compute MD5 checksum in chunks."""
    h = hashlib.md5()
    print(f"Computing MD5 checksum for {path} ({path.stat().st_size} bytes)...")
    t0 = time.time()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    elapsed = time.time() - t0
    md5_hex = h.hexdigest()
    print(f"MD5: {md5_hex} (computed in {elapsed:.1f} s)")
    return md5_hex


def verify_extracted_reflex(target_dir: Path) -> Dict[str, Any]:
    """Verify all 5 strategy folders and 55 participant folders contain the 10 required files."""
    dataset_base = target_dir / "Dataset" if (target_dir / "Dataset").is_dir() else target_dir

    strat_counts: Dict[str, int] = {}
    verified_participants: List[str] = []

    for strat in STRATEGY_FOLDERS:
        strat_dir = dataset_base / strat
        if not strat_dir.is_dir():
            raise FileNotFoundError(f"Missing required strategy directory: {strat_dir}")

        prefix = strat.split("-")[0]
        # Check all 11 participants for this strategy
        found_p = 0
        for p_idx in range(1, 12):
            p_name = f"{prefix}-{p_idx}"
            p_dir = strat_dir / p_name
            if not p_dir.is_dir():
                raise FileNotFoundError(f"Missing participant directory: {p_dir}")

            for req_file in REQUIRED_FILES:
                file_path = p_dir / req_file
                if not file_path.is_file():
                    raise FileNotFoundError(f"Missing required file for participant {p_name}: {file_path}")

            found_p += 1
            verified_participants.append(p_name)

        strat_counts[strat] = found_p

    print(f"Verification OK: found all 5 strategies and {len(verified_participants)} participants:")
    for s, c in strat_counts.items():
        print(f"  {s}: {c} participants (10/10 files present)")

    return {
        "n_strategies": len(STRATEGY_FOLDERS),
        "n_participants": len(verified_participants),
        "strategy_counts": strat_counts,
        "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def update_progress(progress_path: Optional[Path], progress_items: List[Any]) -> None:
    if progress_path is None:
        return
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = progress_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(progress_items), encoding="utf-8")
        tmp_path.replace(progress_path)
    except Exception as e:
        print(f"Warning: failed to write progress to {progress_path}: {e}")


def download_reflex_archive(
    zip_path: Path,
    progress_path: Optional[Path] = None,
) -> None:
    """Download Dataset.zip sequentially using curl with resume and rate-limit retry."""
    print(f"Downloading REFLEX Dataset.zip from Zenodo ({EXPECTED_ZIP_BYTES} bytes)...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Use single sequential stream with curl
    cmd = [
        "curl",
        "-L",
        "-C", "-",
        "--retry", "10",
        "--retry-delay", "15",
        "--retry-all-errors",
        "-o", str(zip_path),
        ZENODO_URL,
    ]

    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        time.sleep(2)
        if zip_path.exists():
            cur_bytes = zip_path.stat().st_size
            cur_mb = cur_bytes // (10 * 1024 * 1024)
            # len(d) in progress.json reflects completed 10MB chunks
            update_progress(progress_path, list(range(cur_mb)))

    if proc.returncode != 0:
        raise RuntimeError(f"curl exited with code {proc.returncode} downloading {ZENODO_URL}")

    # Final byte size check
    final_bytes = zip_path.stat().st_size
    if final_bytes != EXPECTED_ZIP_BYTES:
        raise ValueError(
            f"Downloaded zip size mismatch: {final_bytes} bytes != expected {EXPECTED_ZIP_BYTES} bytes"
        )
    print(f"Download complete: {final_bytes} bytes match expected {EXPECTED_ZIP_BYTES}")


def extract_reflex_archive(
    zip_path: Path,
    target_dir: Path,
    progress_path: Optional[Path] = None,
) -> None:
    """Extract Dataset.zip into target_dir skipping __MACOSX and ._* AppleDouble files."""
    print(f"Extracting {zip_path} into {target_dir}...")
    t0 = time.time()
    n_extracted = 0
    base_progress = EXPECTED_ZIP_BYTES // (10 * 1024 * 1024)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_members = len(members)
        print(f"Total zip members: {total_members}")

        for i, member in enumerate(members):
            filename = member.filename
            if filename.startswith("__MACOSX/") or Path(filename).name.startswith("._"):
                continue

            zf.extract(member, target_dir)
            n_extracted += 1

            if n_extracted % 200 == 0 or i == total_members - 1:
                elapsed = time.time() - t0
                print(f"Extracted {n_extracted}/{total_members} files ({elapsed:.1f} s)...")
                update_progress(progress_path, list(range(base_progress + n_extracted)))

    elapsed = time.time() - t0
    print(f"Extraction complete: {n_extracted} files extracted in {elapsed:.1f} s")


def download_reflex(
    target_dir: Optional[Path] = None,
    verify_only: bool = False,
    force: bool = False,
    progress_file: Optional[Path] = None,
) -> Dict[str, Any]:
    t_start = time.time()
    dest = target_dir or (DATA_ROOT / "raw" / "reflex")
    dest.mkdir(parents=True, exist_ok=True)
    download_json = dest / "DOWNLOAD.json"
    zip_path = dest / "Dataset.zip"
    prog_path = progress_file or (dest / "progress.json")

    # Check idempotency
    if not force and download_json.is_file():
        try:
            meta = json.loads(download_json.read_text(encoding="utf-8"))
            if meta.get("verified") and meta.get("n_participants") == N_PARTICIPANTS:
                # Verify disk state
                verify_extracted_reflex(dest)
                print(f"REFLEX already downloaded and verified in {dest}. Exiting 0.")
                return meta
        except Exception as e:
            print(f"Existing DOWNLOAD.json invalid or verification failed ({e}). Re-running...")

    check_disk_space(dest)

    if not verify_only:
        # Check if zip already exists with exact size
        if not zip_path.is_file() or zip_path.stat().st_size != EXPECTED_ZIP_BYTES:
            download_reflex_archive(zip_path, progress_path=prog_path)

        # Check MD5
        md5_act = file_md5(zip_path)
        if md5_act != EXPECTED_ZIP_MD5:
            raise ValueError(f"MD5 checksum mismatch for {zip_path}: {md5_act} != {EXPECTED_ZIP_MD5}")

        # Extract
        extract_reflex_archive(zip_path, dest, progress_path=prog_path)

    # Verify extracted structure
    ver_info = verify_extracted_reflex(dest)

    # Delete archive to reclaim space
    archive_deleted = False
    if zip_path.is_file():
        print(f"Deleting archive {zip_path} to reclaim 37.22 GB...")
        zip_path.unlink()
        archive_deleted = True

    # Write DOWNLOAD.json
    download_meta = {
        "zenodo_record": ZENODO_RECORD,
        "doi": DOI,
        "license": LICENSE,
        "zip_bytes": EXPECTED_ZIP_BYTES,
        "zip_md5": EXPECTED_ZIP_MD5,
        "n_participants": ver_info["n_participants"],
        "strategy_counts": ver_info["strategy_counts"],
        "verified": True,
        "archive_deleted": archive_deleted,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    download_json.write_text(json.dumps(download_meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {download_json}")
    print(f"items_in={ver_info['n_participants']} items_out={ver_info['n_participants']} excluded=0 elapsed_s={time.time() - t_start:.1f}")

    return download_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify REFLEX dataset for Wave B")
    parser.add_argument("--target-dir", type=Path, default=None, help="Target download directory")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing extracted files")
    parser.add_argument("--force", action="store_true", help="Force re-download / re-extraction")
    parser.add_argument("--progress-file", type=Path, default=None, help="Path to progress.json")
    args = parser.parse_args()

    download_reflex(
        target_dir=args.target_dir,
        verify_only=args.verify_only,
        force=args.force,
        progress_file=args.progress_file,
    )


if __name__ == "__main__":
    main()
