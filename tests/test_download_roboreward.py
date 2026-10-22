from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from tools.download_roboreward import (
    check_disk_space,
    download_roboreward,
    verify_download,
)


@pytest.fixture
def mock_dataset_fixture(tmp_path: Path):
    """Create a local 5-file fixture matching Hub dataset structure."""
    raw_dir = tmp_path / "raw_roboreward"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files_info = [
        ("train/metadata.jsonl", b'{"id": "train_1"}\n'),
        ("val/metadata.jsonl", b'{"id": "val_1"}\n'),
        ("test/metadata.jsonl", b'{"id": "test_1"}\n'),
        ("train/video_1.mp4", b"fake mp4 video bytes 1"),
        ("test/video_2.mp4", b"fake mp4 video bytes 2"),
    ]

    siblings = []
    for rel_path, content in files_info:
        p = raw_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        siblings.append(SimpleNamespace(rfilename=rel_path, size=len(content)))

    mock_info = SimpleNamespace(siblings=siblings)
    mock_api = MagicMock()
    mock_api.dataset_info.return_value = mock_info

    return raw_dir, mock_api, files_info


def test_download_roboreward_complete_passes(mock_dataset_fixture, monkeypatch):
    raw_dir, mock_api, _ = mock_dataset_fixture

    # Fake free disk space: plenty of space
    fake_usage = SimpleNamespace(total=500 * (1024**3), used=100 * (1024**3), free=400 * (1024**3))
    monkeypatch.setattr("shutil.disk_usage", lambda _: fake_usage)

    download_json = download_roboreward(
        target_dir=raw_dir,
        verify_only=True,
        api=mock_api,
    )
    assert download_json.exists()
    meta = json.loads(download_json.read_text(encoding="utf-8"))
    assert meta["n_files"] == 5
    assert meta["n_mp4"] == 2
    assert "train" in meta["metadata_sha256"]
    assert "val" in meta["metadata_sha256"]
    assert "test" in meta["metadata_sha256"]


def test_download_roboreward_missing_file_fails(mock_dataset_fixture, monkeypatch):
    raw_dir, mock_api, _ = mock_dataset_fixture

    fake_usage = SimpleNamespace(total=500 * (1024**3), used=100 * (1024**3), free=400 * (1024**3))
    monkeypatch.setattr("shutil.disk_usage", lambda _: fake_usage)

    # Delete one file
    (raw_dir / "test/video_2.mp4").unlink()

    with pytest.raises(FileNotFoundError, match="Missing file from dataset"):
        verify_download(raw_dir, api=mock_api)


def test_download_roboreward_truncated_file_fails(mock_dataset_fixture, monkeypatch):
    raw_dir, mock_api, _ = mock_dataset_fixture

    fake_usage = SimpleNamespace(total=500 * (1024**3), used=100 * (1024**3), free=400 * (1024**3))
    monkeypatch.setattr("shutil.disk_usage", lambda _: fake_usage)

    # Truncate one file
    (raw_dir / "train/video_1.mp4").write_bytes(b"short")

    with pytest.raises(ValueError, match="File size mismatch"):
        verify_download(raw_dir, api=mock_api)


def test_download_roboreward_insufficient_disk_fails(tmp_path: Path, monkeypatch):
    # Fake free disk space: only 10 GB free (less than dataset bytes + 50 GiB)
    fake_usage = SimpleNamespace(total=100 * (1024**3), used=90 * (1024**3), free=10 * (1024**3))
    monkeypatch.setattr("shutil.disk_usage", lambda _: fake_usage)

    with pytest.raises(RuntimeError, match="Disk rule violated: free space"):
        check_disk_space(tmp_path)
