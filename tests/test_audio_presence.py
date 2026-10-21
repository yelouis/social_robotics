from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.check_audio_presence import (  # noqa: E402
    AUDIBLE_DB,
    compute_segment_dbfs,
    measure_audio_presence_from_samples,
    run_all_sessions_audibility,
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


def test_audio_presence_all_sessions_synthetic(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    holo_dir = tmp_path / "holoassist"
    holo_dir.mkdir()

    # Create 4 sessions:
    # 1. No instructor utterance
    # 2. No silent gap >= 2s
    # 3. +10 dB instructor presence (audible)
    # 4. Flat noise (inaudible)
    sessions = [
        {
            "video_name": "sess_1_no_inst",
            "taskType": "task_a",
            "videoMetadata": {"duration": {"seconds": 10.0}},
            "events": [
                {
                    "label": "Conversation",
                    "start": 1.0,
                    "end": 3.0,
                    "attributes": {"Conversation Purpose": "student-start-conversation_query"},
                }
            ],
        },
        {
            "video_name": "sess_2_no_gap",
            "taskType": "task_b",
            "videoMetadata": {"duration": {"seconds": 3.0}},
            "events": [
                {
                    "label": "Conversation",
                    "start": 0.0,
                    "end": 2.5,
                    "attributes": {"Conversation Purpose": "instructor-start-conversation_guide"},
                }
            ],
        },
        {
            "video_name": "sess_3_audible",
            "taskType": "task_c",
            "videoMetadata": {"duration": {"seconds": 10.0}},
            "events": [
                {
                    "label": "Conversation",
                    "start": 1.0,
                    "end": 4.0,
                    "attributes": {"Conversation Purpose": "instructor-start-conversation_guide"},
                }
            ],
        },
        {
            "video_name": "sess_4_inaudible",
            "taskType": "task_d",
            "videoMetadata": {"duration": {"seconds": 10.0}},
            "events": [
                {
                    "label": "Conversation",
                    "start": 1.0,
                    "end": 4.0,
                    "attributes": {"Conversation Purpose": "instructor-start-conversation_guide"},
                }
            ],
        },
    ]

    ann_file = labels_dir / "data-annotation-trainval-v1_1.json"
    import json
    import soundfile as sf
    import subprocess

    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(sessions, f)

    # Create mock video files for all 4 sessions
    sr = 16000
    rng = np.random.default_rng(0)

    for s in sessions:
        vname = s["video_name"]
        vdir = holo_dir / "videos" / vname / "Export_py"
        vdir.mkdir(parents=True)
        vpath = vdir / "Video_pitchshift.mp4"

        dur = float(s["videoMetadata"]["duration"]["seconds"])
        nsamples = int(dur * sr)
        if vname == "sess_3_audible":
            audio = rng.normal(0, 0.01, size=nsamples).astype(np.float32)
            # Add strong signal in [1.0, 4.0]
            audio[int(1.0 * sr) : int(4.0 * sr)] = rng.normal(0, 0.1, size=int(3.0 * sr)).astype(np.float32)
        else:
            audio = rng.normal(0, 0.03, size=nsamples).astype(np.float32)

        # Write wav then encode to mp4
        wav_tmp = tmp_path / f"{vname}.wav"
        sf.write(str(wav_tmp), audio, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_tmp), "-vn", "-c:a", "aac", str(vpath)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav_tmp.unlink()

    out_file = tmp_path / "runs" / "holoassist_session_audibility.jsonl"
    prog_file = tmp_path / "runs" / "holoassist_session_audibility.progress.json"

    res = run_all_sessions_audibility(
        labels_dir=labels_dir,
        holoassist_dir=holo_dir,
        output_path=out_file,
        progress_path=prog_file,
    )

    assert res["items_in"] == 4
    assert res["audible"] == 1
    assert out_file.exists()
    assert prog_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 4
    by_name = {line_rec["video_name"]: line_rec for line_rec in lines}

    # 1. No instructor utterance
    assert by_name["sess_1_no_inst"]["audible"] is None
    assert by_name["sess_1_no_inst"]["diff_db"] is None
    assert by_name["sess_1_no_inst"]["reason"] == "no instructor utterance"
    assert by_name["sess_1_no_inst"]["n_instructor_spans"] == 0

    # 2. No silent gap >= 2s
    assert by_name["sess_2_no_gap"]["audible"] is None
    assert by_name["sess_2_no_gap"]["diff_db"] is None
    assert by_name["sess_2_no_gap"]["reason"] == "no silent gap >= 2 s"
    assert by_name["sess_2_no_gap"]["n_silent_gaps"] == 0

    # 3. Audible session
    assert by_name["sess_3_audible"]["audible"] is True
    assert by_name["sess_3_audible"]["diff_db"] >= AUDIBLE_DB
    assert by_name["sess_3_audible"]["reason"] is None
    assert by_name["sess_3_audible"]["n_instructor_spans"] == 1

    # 4. Inaudible session
    assert by_name["sess_4_inaudible"]["audible"] is False
    assert by_name["sess_4_inaudible"]["diff_db"] < AUDIBLE_DB
    assert by_name["sess_4_inaudible"]["reason"] is None


def test_audio_presence_determinism_20_sessions():
    """Verify 20 reference sessions from A8 match diff_db to 0.01 dB."""
    from config import DATA_ROOT
    import json

    a8_path = DATA_ROOT / "runs" / "holoassist_audio_presence.json"
    if not a8_path.exists():
        return

    with open(a8_path, "r", encoding="utf-8") as f:
        a8_data = json.load(f)

    all_path = DATA_ROOT / "runs" / "holoassist_session_audibility.jsonl"
    if not all_path.exists():
        return

    all_records = {}
    with open(all_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                all_records[r["video_name"]] = r

    expected_audible = {"z129", "z094", "z195", "z167", "z124-aug-10-22-marius_disassemble", "z035"}

    for a8_sess in a8_data["per_session"]:
        vname = a8_sess["video_name"]
        assert vname in all_records, f"{vname} missing from session audibility results"
        rec = all_records[vname]

        # diff_db matches to 0.01 dB
        assert abs(rec["diff_db"] - a8_sess["diff_db"]) <= 0.015, (
            f"{vname}: diff_db mismatch {rec['diff_db']} vs {a8_sess['diff_db']}"
        )

        is_aud = any(p in vname for p in expected_audible)
        assert rec["audible"] is is_aud, f"{vname}: audible expected {is_aud}, got {rec['audible']}"

