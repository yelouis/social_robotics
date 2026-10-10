from __future__ import annotations

import json
from pathlib import Path
import subprocess
import numpy as np
import pytest

from features.audio import NonverbalAudioEncoder
from features.cache import FeatureCache, sanitize_item_id
from features.visual import FrameEncoder
from harness.items import Item, write_items


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Fast tests
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path: Path):
    cache = FeatureCache(dataset="test_ds", encoder_id="test_enc", data_root=tmp_path)
    item_id = "test:1:0"
    arr = np.random.randn(768).astype(np.float32)

    assert not cache.has(item_id)
    cache.save(item_id, arr, window=[1.0, 3.0], elapsed_ms=12.5)

    assert cache.has(item_id)
    loaded = cache.load(item_id)
    np.testing.assert_allclose(loaded, arr)
    assert loaded.dtype == np.float32

    # Check index.jsonl
    assert cache.index_path.exists()
    lines = [json.loads(line) for line in cache.index_path.read_text().splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["item_id"] == item_id
    assert lines[0]["shape"] == [768]
    assert lines[0]["window"] == [1.0, 3.0]


def test_cache_sanitization(tmp_path: Path):
    cache = FeatureCache(dataset="test_ds", encoder_id="test_enc", data_root=tmp_path)
    dirty_id = "oops:video 1/dir#2:0"
    clean_id = sanitize_item_id(dirty_id)
    assert clean_id == "oops_video_1_dir_2_0"
    p = cache.path(dirty_id)
    assert p.name == "oops_video_1_dir_2_0.npy"


def test_failed_extraction_writes_no_npy(tmp_path: Path, monkeypatch):
    from features.extract import run_extraction
    monkeypatch.setattr("features.extract.DATA_ROOT", tmp_path)

    items_dir = tmp_path / "items" / "test_ds"
    items_dir.mkdir(parents=True, exist_ok=True)
    item = Item("test:fail:0", "test_ds", "g1", "/nonexistent.mp4", [0.0, 1.0], [1.0, 2.0], 1, "s", "c")
    write_items([item], items_dir / "items.jsonl")

    # Run extraction (will fail because file nonexistent)
    run_extraction(dataset="test_ds", encoder_id="siglip-b16-224")

    cache = FeatureCache(dataset="test_ds", encoder_id="siglip-b16-224", data_root=tmp_path)
    assert not cache.has("test:fail:0")
    errors_path = cache.cache_dir / "errors.jsonl"
    assert errors_path.exists()
    lines = [json.loads(line) for line in errors_path.read_text().splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["item_id"] == "test:fail:0"


def test_resumability(tmp_path: Path, monkeypatch):
    from features.extract import run_extraction
    monkeypatch.setattr("features.extract.DATA_ROOT", tmp_path)

    items_dir = tmp_path / "items" / "test_ds"
    items_dir.mkdir(parents=True, exist_ok=True)
    items = [
        Item(f"test:{i}:0", "test_ds", f"g{i}", "/v", [0.0, 1.0], [1.0, 2.0], 1, "s", "c")
        for i in range(6)
    ]
    write_items(items, items_dir / "items.jsonl")

    cache = FeatureCache(dataset="test_ds", encoder_id="siglip-b16-224", data_root=tmp_path)

    # Mock encoder: fails on item index 2 (the 3rd item)
    call_count = [0]
    processed_items = []

    class MockEncoder:
        encoder_id = "siglip-b16-224"

        def encode_window(self, video_path, window):
            curr_call = call_count[0]
            call_count[0] += 1
            if curr_call == 2:
                raise SystemExit("Injected crash on 3rd item")
            processed_items.append(curr_call)
            return np.ones(768, dtype=np.float32)

    monkeypatch.setattr("features.extract.FrameEncoder", MockEncoder)

    # First run: crashes on 3rd item
    with pytest.raises(SystemExit):
        run_extraction(dataset="test_ds", encoder_id="siglip-b16-224")
    assert cache.has("test:0:0")
    assert cache.has("test:1:0")
    assert not cache.has("test:2:0")

    # Second run without injected failure: should only process remaining items
    processed_on_rerun = []

    class NormalMockEncoder:
        encoder_id = "siglip-b16-224"

        def encode_window(self, video_path, window):
            processed_on_rerun.append(video_path)
            return np.ones(768, dtype=np.float32)

    monkeypatch.setattr("features.extract.FrameEncoder", NormalMockEncoder)
    run_extraction(dataset="test_ds", encoder_id="siglip-b16-224")

    # The rerun should only process items 2, 3, 4, 5 (4 items)
    assert len(processed_on_rerun) == 4
    for i in range(6):
        assert cache.has(f"test:{i}:0")


# ---------------------------------------------------------------------------
# Slow tests (models, audio, video)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audio_fixtures(tmp_path_factory):
    fn_dir = tmp_path_factory.mktemp("audio_data")
    sam_aiff = fn_dir / "sam.aiff"
    dan_aiff = fn_dir / "dan.aiff"
    sentence = "Stop doing that right now, you are making a terrible mistake!"
    subprocess.run(["say", "-v", "Samantha", "-o", str(sam_aiff), sentence], check=True)
    subprocess.run(["say", "-v", "Daniel", "-o", str(dan_aiff), sentence], check=True)

    sam_wav = fn_dir / "sam.wav"
    dan_wav = fn_dir / "dan.wav"
    silence_wav = fn_dir / "silence.wav"

    subprocess.run(["ffmpeg", "-y", "-i", str(sam_aiff), "-ar", "16000", "-ac", "1", "-t", "3.0", str(sam_wav)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(dan_aiff), "-ar", "16000", "-ac", "1", "-t", "3.0", str(dan_wav)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3.0", str(silence_wav)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return sam_wav, dan_wav, silence_wav


@pytest.fixture(scope="module")
def video_fixtures(tmp_path_factory):
    fn_dir = tmp_path_factory.mktemp("video_data")
    real_mp4 = fn_dir / "real.mp4"
    black_mp4 = fn_dir / "black.mp4"

    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(real_mp4)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:duration=5:size=320x240:rate=10",
        "-pix_fmt", "yuv420p", str(black_mp4)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return real_mp4, black_mp4


@pytest.mark.slow
def test_shapes_dtypes_and_finite(audio_fixtures, video_fixtures):
    sam_wav, _, _ = audio_fixtures
    real_mp4, _ = video_fixtures

    v_enc = FrameEncoder()
    v_feat = v_enc.encode_window(real_mp4, [1.0, 3.0])
    assert v_feat.shape == (768,)
    assert v_feat.dtype == np.float32
    assert np.all(np.isfinite(v_feat))

    a_enc = NonverbalAudioEncoder()
    a_feat = a_enc.encode_wav(sam_wav)
    assert a_feat.shape == (1024,)
    assert a_feat.dtype == np.float32
    assert np.all(np.isfinite(a_feat))


@pytest.mark.slow
def test_determinism(audio_fixtures, video_fixtures):
    sam_wav, _, _ = audio_fixtures
    real_mp4, _ = video_fixtures

    v_enc = FrameEncoder()
    v_feat1 = v_enc.encode_window(real_mp4, [1.0, 3.0])
    v_feat2 = v_enc.encode_window(real_mp4, [1.0, 3.0])
    assert float(np.max(np.abs(v_feat1 - v_feat2))) <= 1e-5

    a_enc = NonverbalAudioEncoder()
    a_feat1 = a_enc.encode_wav(sam_wav)
    a_feat2 = a_enc.encode_wav(sam_wav)
    assert float(np.max(np.abs(a_feat1 - a_feat2))) <= 1e-5


@pytest.mark.slow
def test_encoders_carry_information(audio_fixtures, video_fixtures):
    sam_wav, dan_wav, silence_wav = audio_fixtures
    real_mp4, black_mp4 = video_fixtures

    # Audio encoder: cos(speechA, speechB) > cos(speechA, silence)
    a_enc = NonverbalAudioEncoder()
    feat_sam = a_enc.encode_wav(sam_wav)
    feat_dan = a_enc.encode_wav(dan_wav)
    feat_sil = a_enc.encode_wav(silence_wav)

    cos_speech = cos_sim(feat_sam, feat_dan)
    cos_silence = cos_sim(feat_sam, feat_sil)
    assert cos_speech > cos_silence, f"Expected {cos_speech} > {cos_silence}"

    # Visual encoder: cos(real window, black window) < cos(same real window, shifted by +0.5 s)
    v_enc = FrameEncoder()
    feat_real = v_enc.encode_window(real_mp4, [1.0, 3.0])
    feat_black = v_enc.encode_window(black_mp4, [1.0, 3.0])
    feat_shifted = v_enc.encode_window(real_mp4, [1.5, 3.5])

    cos_black = cos_sim(feat_real, feat_black)
    cos_shifted = cos_sim(feat_real, feat_shifted)
    assert cos_black < cos_shifted, f"Expected {cos_black} < {cos_shifted}"
