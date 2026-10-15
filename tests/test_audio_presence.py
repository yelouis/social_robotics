from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.check_audio_presence import (  # noqa: E402
    compute_segment_dbfs,
    measure_audio_presence_from_samples,
)


def test_compute_segment_dbfs_no_sentinel():
    audio = np.ones(16000, dtype=np.float32)
    # Valid span
    val = compute_segment_dbfs(audio, 0.0, 1.0, sr=16000)
    assert val is not None
    assert np.isclose(val, 0.0, atol=1e-3)

    # Empty span (no samples) -> returns None, not -100.0 sentinel
    assert compute_segment_dbfs(audio, 1.0, 1.0, sr=16000) is None
    assert compute_segment_dbfs(audio, 2.0, 1.0, sr=16000) is None
    assert compute_segment_dbfs(audio, 2.0, 3.0, sr=16000) is None


def test_audio_presence_falsification():
    sr = 16000
    duration = 10.0
    total_samples = int(duration * sr)

    events = [
        {
            "label": "Conversation",
            "start": 1.0,
            "end": 3.0,
            "attributes": {"Conversation Purpose": "instructor-start-conversation_guide"},
        },
        # Silent gaps will be [0.0, 1.0] (1.0s, filtered by min_gap_s=2.0) and [3.0, 10.0] (7.0s, kept)
    ]

    # 1. Test case: Instructor spans +10 dB over silent gap (expected: PASS)
    rng = np.random.default_rng(0)
    # Background noise at -30 dBFS (amplitude ~0.0316)
    noise_bg = rng.normal(0, 0.0316, size=total_samples).astype(np.float32)

    # Add signal to instructor span [1.0, 3.0] s to be +10 dB higher (amplitude ~0.1)
    s_idx = int(1.0 * sr)
    e_idx = int(3.0 * sr)
    noise_bg[s_idx:e_idx] = rng.normal(0, 0.1, size=(e_idx - s_idx)).astype(np.float32)

    res_pass = measure_audio_presence_from_samples(
        audio=noise_bg,
        sr=sr,
        events=events,
        duration=duration,
        video_name="test_speech",
    )
    assert res_pass["diff_db"] >= 3.5, f"Expected diff_db >= 3.5, got {res_pass['diff_db']}"
    assert np.isclose(res_pass["diff_db"], 10.0, atol=2.0)

    # 2. Test case: Flat noise (equal energy across spans) -> falsification (expected: FAIL)
    flat_noise = rng.normal(0, 0.05, size=total_samples).astype(np.float32)

    res_fail = measure_audio_presence_from_samples(
        audio=flat_noise,
        sr=sr,
        events=events,
        duration=duration,
        video_name="test_flat_noise",
    )
    # Flat noise difference should be close to 0 dB (< 3.5 dB requirement)
    assert abs(res_fail["diff_db"]) < 2.0
    assert res_fail["diff_db"] < 3.5, f"Falsification failed: flat noise diff_db {res_fail['diff_db']} >= 3.5"
