from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile

from config import DATA_ROOT
from sources.holoassist import get_holoassist_video_path, load_raw_annotations


def extract_session_audio(video_path: Path, out_wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_wav_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path}: {proc.stderr.decode('utf-8', errors='replace')}"
        )


def compute_segment_dbfs(
    audio: np.ndarray,
    start_s: float,
    end_s: float,
    sr: int = 16000,
) -> Optional[float]:
    """Computes RMS dBFS for a segment. Returns None if segment contains no samples."""
    s_idx = max(0, int(round(start_s * sr)))
    e_idx = min(len(audio), int(round(end_s * sr)))
    if e_idx <= s_idx:
        return None
    seg = audio[s_idx:e_idx]
    rms = float(np.sqrt(np.mean(seg**2)))
    if rms <= 1e-9:
        rms = 1e-9
    return float(20.0 * np.log10(rms))


def get_conversation_spans(
    events: List[Dict[str, Any]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Returns (all_conv_spans, instructor_conv_spans)."""
    all_convs: List[Tuple[float, float]] = []
    inst_convs: List[Tuple[float, float]] = []
    for e in events:
        if e.get("label") == "Conversation":
            st = float(e.get("start", 0.0))
            en = float(e.get("end", 0.0))
            if en > st:
                all_convs.append((st, en))
                attrs = e.get("attributes", {})
                if attrs.get("Conversation Purpose", "").startswith("instructor"):
                    inst_convs.append((st, en))
    return all_convs, inst_convs


def get_silent_gaps(
    all_convs: List[Tuple[float, float]],
    duration: float,
    min_gap_s: float = 2.0,
) -> List[Tuple[float, float]]:
    """Finds silent gaps where no conversation occurs, with length >= min_gap_s."""
    if not all_convs:
        return [(0.0, duration)] if duration >= min_gap_s else []

    sorted_convs = sorted(all_convs, key=lambda x: x[0])
    merged: List[List[float]] = []
    for st, en in sorted_convs:
        if not merged or merged[-1][1] < st:
            merged.append([st, en])
        else:
            merged[-1][1] = max(merged[-1][1], en)

    gaps: List[Tuple[float, float]] = []
    cur = 0.0
    for st, en in merged:
        if st > cur:
            gaps.append((cur, st))
        cur = max(cur, en)
    if cur < duration:
        gaps.append((cur, duration))

    return [(st, en) for st, en in gaps if en - st >= min_gap_s]


def measure_audio_presence_from_samples(
    audio: np.ndarray,
    sr: int,
    events: List[Dict[str, Any]],
    duration: float,
    video_name: str = "synthetic",
) -> Dict[str, Any]:
    """Measures instructor audio presence directly from audio samples."""
    all_convs, inst_convs = get_conversation_spans(events)
    silent_gaps = get_silent_gaps(all_convs, duration, min_gap_s=2.0)

    inst_dbfs_list: List[float] = []
    n_excluded_inst = 0
    for st, en in inst_convs:
        val = compute_segment_dbfs(audio, st, en, sr)
        if val is not None:
            inst_dbfs_list.append(val)
        else:
            n_excluded_inst += 1

    silent_dbfs_list: List[float] = []
    n_excluded_silent = 0
    for st, en in silent_gaps:
        val = compute_segment_dbfs(audio, st, en, sr)
        if val is not None:
            silent_dbfs_list.append(val)
        else:
            n_excluded_silent += 1

    if not inst_dbfs_list or not silent_dbfs_list:
        raise ValueError(
            f"Cannot compute audio presence for {video_name}: "
            f"valid instructor spans={len(inst_dbfs_list)}, valid silent gaps={len(silent_dbfs_list)}"
        )

    mean_inst_dbfs = float(np.mean(inst_dbfs_list))
    mean_silent_dbfs = float(np.mean(silent_dbfs_list))
    diff_db = mean_inst_dbfs - mean_silent_dbfs

    return {
        "video_name": video_name,
        "duration": duration,
        "n_instructor_utterances": len(inst_convs),
        "n_excluded_instructor_spans": n_excluded_inst,
        "mean_instructor_dbfs": mean_inst_dbfs,
        "n_silent_gaps": len(silent_gaps),
        "n_excluded_silent_gaps": n_excluded_silent,
        "mean_silent_dbfs": mean_silent_dbfs,
        "diff_db": diff_db,
    }


def measure_session_audio_presence(
    session: Dict[str, Any],
    video_path: Path,
) -> Dict[str, Any]:
    vname = session.get("video_name", "")
    events = session.get("events", [])
    duration = float(session.get("videoMetadata", {}).get("duration", {}).get("seconds", 0.0))

    with tempfile.NamedTemporaryFile("wb", suffix=".wav", dir=tempfile.gettempdir(), delete=False) as tf:
        temp_wav = Path(tf.name)

    try:
        extract_session_audio(video_path, temp_wav)
        audio, sr = soundfile.read(temp_wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        actual_duration = len(audio) / sr
        if duration <= 0:
            duration = actual_duration

        return measure_audio_presence_from_samples(
            audio=audio,
            sr=sr,
            events=events,
            duration=duration,
            video_name=vname,
        )
    finally:
        if temp_wav.exists():
            temp_wav.unlink()


def run_audio_presence_check(
    n_sessions: int = 20,
    seed: int = 0,
    labels_dir: Optional[Path] = None,
    holoassist_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    sessions = load_raw_annotations(labels_dir)
    sessions = sorted(sessions, key=lambda s: s.get("video_name", ""))

    # Pre-filter candidates: must have >=1 instructor utterance and >=1 silent gap >=2s, and existing video
    candidates: List[Tuple[Dict[str, Any], Path]] = []
    for s in sessions:
        vname = s.get("video_name", "")
        vpath = get_holoassist_video_path(vname, holoassist_dir)
        if not vpath.exists():
            continue
        events = s.get("events", [])
        duration = float(s.get("videoMetadata", {}).get("duration", {}).get("seconds", 0.0))
        all_convs, inst_convs = get_conversation_spans(events)
        silent_gaps = get_silent_gaps(all_convs, duration, min_gap_s=2.0)
        if len(inst_convs) >= 1 and len(silent_gaps) >= 1:
            candidates.append((s, vpath))

    if not candidates:
        raise RuntimeError("No candidate sessions found with instructor utterances, silent gaps, and existing videos")

    rng = np.random.default_rng(seed)
    n_to_select = min(n_sessions, len(candidates))
    indices = rng.choice(len(candidates), size=n_to_select, replace=False)
    selected_candidates = [candidates[i] for i in indices]

    results = []
    for idx, (sess, vpath) in enumerate(selected_candidates, 1):
        vname = sess.get("video_name", "")
        print(f"[{idx}/{n_to_select}] Measuring audio presence for {vname}...")
        res = measure_session_audio_presence(sess, vpath)
        print(
            f"    Instructor: {res['mean_instructor_dbfs']:.2f} dBFS, "
            f"Silent: {res['mean_silent_dbfs']:.2f} dBFS -> Diff: {res['diff_db']:+.2f} dB"
        )
        results.append(res)

    diffs = [r["diff_db"] for r in results]
    median_diff = float(np.median(diffs))
    passed = bool(median_diff >= 3.5)

    summary = {
        "n_sessions": len(results),
        "seed": seed,
        "per_session": results,
        "median_diff_db": median_diff,
        "required_diff_db": 3.5,
        "passed": passed,
    }

    out_file = output_path or (DATA_ROOT / "runs" / "holoassist_audio_presence.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="HoloAssist instructor audio presence check (§A8.2)")
    parser.add_argument("--n-sessions", type=int, default=20, help="Number of sessions to measure")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for session selection")
    parser.add_argument("--output", type=Path, default=None, help="Output summary path")
    args = parser.parse_args()

    summary = run_audio_presence_check(
        n_sessions=args.n_sessions,
        seed=args.seed,
        output_path=args.output,
    )

    print("\n--- Audio Presence Summary ---")
    print(f"Sessions measured: {summary['n_sessions']}")
    print(f"Median per-session difference: {summary['median_diff_db']:+.2f} dB (requirement: >= +3.5 dB)")
    print(f"Verdict: {'PASS' if summary['passed'] else 'FAIL'}")

    if not summary["passed"]:
        raise SystemExit("Audio presence check FAILED: median difference < +3.5 dB")


if __name__ == "__main__":
    main()
