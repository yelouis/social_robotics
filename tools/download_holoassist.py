from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from config import DATA_ROOT


HOLOASSIST_VIDEO_URL = "https://hl2data.z5.web.core.windows.net/holoassist-data-release/video_pitch_shifted.tar"
EXPECTED_BYTES = 197783674880


def update_progress(holoassist_dir: Path, step_id: str) -> None:
    progress_file = holoassist_dir / "progress.json"
    steps = []
    if progress_file.exists():
        try:
            steps = json.loads(progress_file.read_text(encoding="utf-8"))
        except Exception:
            steps = []
    if step_id not in steps:
        steps.append(step_id)
    with tempfile.NamedTemporaryFile("w", dir=holoassist_dir, delete=False, encoding="utf-8") as tf:
        json.dump(steps, tf, indent=2)
        temp_name = tf.name
        tf.flush()
        os.fsync(tf.fileno())
    os.replace(temp_name, progress_file)


def check_disk_space(target_dir: Path, required_gb: float = 50.0) -> None:
    usage = shutil.disk_usage(target_dir)
    free_gb = usage.free / (1024**3)
    if free_gb < required_gb:
        raise RuntimeError(f"Disk rule violated: free space {free_gb:.1f} GiB < {required_gb} GiB")
    print(f"Disk check OK: {free_gb:.1f} GiB free")


def compute_sha256(path: Path) -> str:
    print(f"Computing sha256 for {path}...")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def update_download_json(download_json_path: Path, entry: dict) -> None:
    records = []
    if download_json_path.exists():
        try:
            records = json.loads(download_json_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = [records]
        except Exception:
            records = []

    # Update matching filename or append
    updated = False
    for i, r in enumerate(records):
        if r.get("filename") == entry.get("filename"):
            records[i] = entry
            updated = True
            break
    if not updated:
        records.append(entry)

    download_json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    holoassist_dir = DATA_ROOT / "raw" / "holoassist"
    holoassist_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = holoassist_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    archive_path = holoassist_dir / "video_pitch_shifted.tar"
    download_json_path = holoassist_dir / "DOWNLOAD.json"
    extracted_flag = videos_dir / ".extracted"

    check_disk_space(holoassist_dir, required_gb=50.0)

    if extracted_flag.exists():
        vid_count = len(list(videos_dir.rglob("*.mp4")))
        print(f"HoloAssist pitch-shifted videos already extracted: {vid_count} mp4 files found.")
        update_progress(holoassist_dir, "verified")
        return

    # 1. Download archive with curl resume support
    if not archive_path.exists() or archive_path.stat().st_size != EXPECTED_BYTES:
        print(f"Downloading HoloAssist pitch-shifted videos from {HOLOASSIST_VIDEO_URL}...")
        update_progress(holoassist_dir, "downloading")
        cmd = [
            "curl", "-L", "-C", "-",
            "--retry", "10",
            "--retry-delay", "5",
            "-o", str(archive_path),
            HOLOASSIST_VIDEO_URL,
        ]
        subprocess.run(cmd, check=True)

    size_bytes = archive_path.stat().st_size
    print(f"Download complete: {size_bytes} bytes (expected {EXPECTED_BYTES})")
    if size_bytes != EXPECTED_BYTES:
        raise RuntimeError(f"Downloaded size {size_bytes} != expected {EXPECTED_BYTES}")

    sha256_hash = compute_sha256(archive_path)
    print(f"Archive sha256: {sha256_hash}")

    download_entry = {
        "url": HOLOASSIST_VIDEO_URL,
        "filename": "video_pitch_shifted.tar",
        "bytes": size_bytes,
        "sha256": sha256_hash,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "extracted": False,
        "archive_deleted": False,
    }
    update_download_json(download_json_path, download_entry)
    update_progress(holoassist_dir, "downloaded")

    # 2. Extract into videos_dir
    print(f"Extracting {archive_path} into {videos_dir}...")
    update_progress(holoassist_dir, "extracting")
    subprocess.run(["tar", "-xf", str(archive_path), "-C", str(videos_dir)], check=True)

    # 3. Verify extracted files
    print("Verifying extracted video files...")
    vid_count = len(list(videos_dir.rglob("*.mp4")))
    print(f"Found {vid_count} mp4 videos in {videos_dir}")
    if vid_count < 1000:
        raise RuntimeError(f"Expected >= 1000 mp4 videos, but found only {vid_count}!")

    # 4. Delete archive to reclaim disk space
    print(f"Deleting archive {archive_path} to reclaim 184.20 GB...")
    archive_path.unlink()
    extracted_flag.write_text("done\n", encoding="utf-8")
    update_progress(holoassist_dir, "extracted")

    download_entry["extracted"] = True
    download_entry["archive_deleted"] = True
    download_entry["archive_deleted_at"] = datetime.now(timezone.utc).isoformat()
    download_entry["extracted_video_count"] = vid_count
    update_download_json(download_json_path, download_entry)
    update_progress(holoassist_dir, "verified")
    print(f"HoloAssist download and extraction complete: {vid_count} videos verified.")


if __name__ == "__main__":
    main()
