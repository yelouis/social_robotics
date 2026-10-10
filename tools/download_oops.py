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


OOPS_URL = "https://oops.cs.columbia.edu/data/video_and_anns.tar.gz"
EXPECTED_BYTES = 47904996151


def update_progress(oops_dir: Path, step_id: str) -> None:
    progress_file = oops_dir / "progress.json"
    steps = []
    if progress_file.exists():
        try:
            steps = json.loads(progress_file.read_text(encoding="utf-8"))
        except Exception:
            steps = []
    if step_id not in steps:
        steps.append(step_id)
    with tempfile.NamedTemporaryFile("w", dir=oops_dir, delete=False, encoding="utf-8") as tf:
        json.dump(steps, tf)
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
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    oops_dir = DATA_ROOT / "raw" / "oops"
    oops_dir.mkdir(parents=True, exist_ok=True)
    archive_path = oops_dir / "video_and_anns.tar.gz"
    download_json_path = oops_dir / "DOWNLOAD.json"

    check_disk_space(oops_dir, required_gb=50.0)

    # 1. Download if not already extracted
    extracted_flag = oops_dir / ".extracted"
    if not extracted_flag.exists():
        if not archive_path.exists() or archive_path.stat().st_size != EXPECTED_BYTES:
            print(f"Downloading Oops! bundle from {OOPS_URL}...")
            cmd = [
                "curl", "-L", "-C", "-",
                "-o", str(archive_path),
                OOPS_URL,
            ]
            subprocess.run(cmd, check=True)

        size_bytes = archive_path.stat().st_size
        print(f"Download complete: {size_bytes} bytes")
        sha256_hash = compute_sha256(archive_path)

        download_meta = {
            "url": OOPS_URL,
            "bytes": size_bytes,
            "sha256": sha256_hash,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "extracted": False,
            "archive_deleted": False,
        }
        download_json_path.write_text(json.dumps(download_meta, indent=2), encoding="utf-8")
        update_progress(oops_dir, "downloaded")

        # 2. Extract
        print(f"Extracting outer {archive_path} into {oops_dir}...")
        subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(oops_dir)], check=True)

        inner_anns = oops_dir / "oops_dataset" / "annotations.tar.gz"
        inner_video = oops_dir / "oops_dataset" / "video.tar.gz"

        if inner_anns.exists():
            print(f"Extracting inner annotations {inner_anns}...")
            subprocess.run(["tar", "-xzf", str(inner_anns), "-C", str(oops_dir / "oops_dataset")], check=True)
            print("Deleting inner annotations archive...")
            inner_anns.unlink()

        if inner_video.exists():
            print(f"Extracting inner video {inner_video}...")
            subprocess.run(["tar", "-xzf", str(inner_video), "-C", str(oops_dir / "oops_dataset")], check=True)
            print("Deleting inner video archive...")
            inner_video.unlink()

        # 3. Verify
        print("Verifying extracted files...")
        # Check that videos and annotations exist
        vid_count = len(list(oops_dir.rglob("*.mp4")))
        json_count = len(list(oops_dir.rglob("*.json")))
        print(f"Found {vid_count} mp4 videos and {json_count} json files")
        if vid_count == 0:
            raise RuntimeError("Extraction produced 0 mp4 videos!")

        # 4. Delete outer archive
        print(f"Deleting outer archive {archive_path} to reclaim space...")
        archive_path.unlink()
        extracted_flag.write_text("done\n", encoding="utf-8")
        update_progress(oops_dir, "extracted")

        download_meta["extracted"] = True
        download_meta["archive_deleted"] = True
        download_meta["archive_deleted_at"] = datetime.now(timezone.utc).isoformat()
        download_meta["extracted_video_count"] = vid_count
        download_json_path.write_text(json.dumps(download_meta, indent=2), encoding="utf-8")
        update_progress(oops_dir, "verified")
    else:
        # If extracted flag exists, make sure inner archives are also unpacked if they were left
        inner_anns = oops_dir / "oops_dataset" / "annotations.tar.gz"
        inner_video = oops_dir / "oops_dataset" / "video.tar.gz"
        if inner_anns.exists() or inner_video.exists():
            if inner_anns.exists():
                print(f"Extracting remaining inner annotations {inner_anns}...")
                subprocess.run(["tar", "-xzf", str(inner_anns), "-C", str(oops_dir / "oops_dataset")], check=True)
                inner_anns.unlink()
            if inner_video.exists():
                print(f"Extracting remaining inner video {inner_video}...")
                subprocess.run(["tar", "-xzf", str(inner_video), "-C", str(oops_dir / "oops_dataset")], check=True)
                inner_video.unlink()
            vid_count = len(list(oops_dir.rglob("*.mp4")))
            print(f"Found {vid_count} mp4 videos")
        print("Oops! is already extracted and verified.")
        update_progress(oops_dir, "verified")


if __name__ == "__main__":
    main()
