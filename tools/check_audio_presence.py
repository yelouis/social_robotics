from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile

from config import DATA_ROOT
from sources.holoassist import get_holoassist_video_path, load_raw_annotations

AUDIBLE_DB: float = 3.5


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


def run_all_sessions_audibility(
    labels_dir: Optional[Path] = None,
    holoassist_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    sessions = load_raw_annotations(labels_dir)
    sessions = sorted(sessions, key=lambda s: s.get("video_name", ""))

    # Filter for sessions whose video exists
    sessions_with_video: List[Tuple[Dict[str, Any], Path]] = []
    for s in sessions:
        vname = s.get("video_name", "")
        vpath = get_holoassist_video_path(vname, holoassist_dir)
        if vpath.exists():
            sessions_with_video.append((s, vpath))

    out_file = output_path or (DATA_ROOT / "runs" / "holoassist_session_audibility.jsonl")
    prog_file = progress_path or (out_file.parent / "holoassist_session_audibility.progress.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    finished_names: List[str] = []
    if prog_file.exists():
        try:
            with open(prog_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, list):
                    finished_names = d
        except Exception:
            finished_names = []

    # Also reconcile with existing jsonl if prog_file was missing or incomplete
    if out_file.exists() and not finished_names:
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        vn = rec.get("video_name")
                        if vn and vn not in finished_names:
                            finished_names.append(vn)
                    except Exception:
                        pass

    finished_set = set(finished_names)
    total_sessions = len(sessions_with_video)

    for idx, (sess, vpath) in enumerate(sessions_with_video, 1):
        vname = sess.get("video_name", "")
        task = sess.get("taskType", "")
        if vname in finished_set:
            continue

        events = sess.get("events", [])
        duration = float(sess.get("videoMetadata", {}).get("duration", {}).get("seconds", 0.0))
        all_convs, inst_convs = get_conversation_spans(events)
        silent_gaps = get_silent_gaps(all_convs, duration, min_gap_s=2.0)

        if len(inst_convs) == 0:
            rec = {
                "video_name": vname,
                "task": task,
                "diff_db": None,
                "audible": None,
                "n_instructor_spans": 0,
                "n_silent_gaps": len(silent_gaps),
                "reason": "no instructor utterance",
            }
        elif len(silent_gaps) == 0:
            rec = {
                "video_name": vname,
                "task": task,
                "diff_db": None,
                "audible": None,
                "n_instructor_spans": len(inst_convs),
                "n_silent_gaps": 0,
                "reason": "no silent gap >= 2 s",
            }
        else:
            try:
                res = measure_session_audio_presence(sess, vpath)
                diff_val = round(float(res["diff_db"]), 2)
                is_aud = bool(diff_val >= AUDIBLE_DB)
                rec = {
                    "video_name": vname,
                    "task": task,
                    "diff_db": diff_val,
                    "audible": is_aud,
                    "n_instructor_spans": len(inst_convs),
                    "n_silent_gaps": len(silent_gaps),
                    "reason": None,
                }
            except Exception as e:
                err_msg = str(e)[:120]
                rec = {
                    "video_name": vname,
                    "task": task,
                    "diff_db": None,
                    "audible": None,
                    "n_instructor_spans": len(inst_convs),
                    "n_silent_gaps": len(silent_gaps),
                    "reason": f"error: {err_msg}",
                }

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

        finished_names.append(vname)
        finished_set.add(vname)

        temp_prog = prog_file.with_name(f".{prog_file.name}.tmp")
        with open(temp_prog, "w", encoding="utf-8") as f:
            json.dump(finished_names, f)
            f.flush()
            os.fsync(f.fileno())
        temp_prog.replace(prog_file)

    all_recs: List[Dict[str, Any]] = []
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_recs.append(json.loads(line))
                    except Exception:
                        pass

    n_aud = sum(1 for r in all_recs if r.get("audible") is True)
    n_inaud = sum(1 for r in all_recs if r.get("audible") is False)
    n_undef = sum(1 for r in all_recs if r.get("audible") is None)
    elapsed_s = time.time() - start_time

    items_in = total_sessions
    items_out = len(all_recs)
    excluded = 0
    print(
        f"items_in={items_in} items_out={items_out} excluded={excluded} "
        f"elapsed_s={elapsed_s:.2f} audible={n_aud} inaudible={n_inaud} undefined={n_undef}"
    )

    return {
        "items_in": items_in,
        "items_out": items_out,
        "excluded": excluded,
        "elapsed_s": elapsed_s,
        "audible": n_aud,
        "inaudible": n_inaud,
        "undefined": n_undef,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HoloAssist instructor audio presence check (§A8.2)")
    parser.add_argument("--n-sessions", type=int, default=20, help="Number of sessions to measure")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for session selection")
    parser.add_argument("--output", type=Path, default=None, help="Output summary path")
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Measure audibility for all annotated sessions with video",
    )
    parser.add_argument("--labels-dir", type=Path, default=None, help="Directory containing raw annotations")
    parser.add_argument("--holoassist-dir", type=Path, default=None, help="Directory containing HoloAssist dataset")
    args = parser.parse_args()

    if args.all_sessions:
        run_all_sessions_audibility(
            labels_dir=args.labels_dir,
            holoassist_dir=args.holoassist_dir,
            output_path=args.output,
        )
        return

    summary = run_audio_presence_check(
        n_sessions=args.n_sessions,
        seed=args.seed,
        output_path=args.output,
        labels_dir=args.labels_dir,
        holoassist_dir=args.holoassist_dir,
    )

    print("\n--- Audio Presence Summary ---")
    print(f"Sessions measured: {summary['n_sessions']}")
    print(f"Median per-session difference: {summary['median_diff_db']:+.2f} dB (requirement: >= +3.5 dB)")
    print(f"Verdict: {'PASS' if summary['passed'] else 'FAIL'}")

    if not summary["passed"]:
        raise SystemExit("Audio presence check FAILED: median difference < +3.5 dB")


if __name__ == "__main__":
    main()
