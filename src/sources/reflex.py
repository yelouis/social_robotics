from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config import DATA_ROOT
from harness.items import Item, validate_items, write_items
from harness.splits import make_group_split


STRATEGY_FOLDERS = (
    "C1-Fixed-Low",
    "C2-Fixed-Medium",
    "C3-Fixed-High",
    "D1-Decay-Smooth",
    "D2-Decay-Rapid",
)


def get_reflex_raw_dir(raw_dir: Optional[Path] = None) -> Path:
    base = raw_dir or (DATA_ROOT / "raw" / "reflex")
    if (base / "Dataset").is_dir():
        return base / "Dataset"
    return base


def get_video_info(video_path: Path) -> Tuple[float, float, int]:
    """Query container fps, duration (seconds), and total frame count via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    info = json.loads(res.stdout)

    stream = info.get("streams", [{}])[0]
    fmt = info.get("format", {})

    # FPS
    r_fps_str = stream.get("r_frame_rate", "")
    avg_fps_str = stream.get("avg_frame_rate", "")
    fps = 0.0
    for s in (r_fps_str, avg_fps_str):
        if s and "/" in s:
            num, den = s.split("/")
            if float(den) > 0:
                fps = float(num) / float(den)
                if fps > 0:
                    break
        elif s:
            fps = float(s)
            if fps > 0:
                break
    if fps <= 0:
        fps = 40.0 / 9.0  # REFLEX standard default

    # Duration
    dur_str = stream.get("duration") or fmt.get("duration")
    duration = float(dur_str) if dur_str else 0.0

    # Number of frames
    nb_frames_str = stream.get("nb_frames")
    nb_frames = int(nb_frames_str) if nb_frames_str and nb_frames_str.isdigit() else int(round(duration * fps))

    return fps, duration, nb_frames


def frame_to_container_seconds(frame_1idx: int, fps: float) -> float:
    """Map 1-indexed frame to container seconds: (frame_1idx - 1) / fps."""
    return round((frame_1idx - 1) / fps, 4)


def discover_participants(reflex_dir: Path) -> List[Tuple[str, str, Path]]:
    """Discover all participant folders: returns list of (strategy, participant_id, dir_path)."""
    participants: List[Tuple[str, str, Path]] = []
    # If reflex_dir contains strategy folders directly
    for strat in STRATEGY_FOLDERS:
        strat_dir = reflex_dir / strat
        if strat_dir.is_dir():
            for child in sorted(strat_dir.iterdir()):
                if child.is_dir() and not child.name.startswith((".", "_")):
                    if (child / "analysis.csv").is_file():
                        participants.append((strat, child.name, child))
    if not participants:
        # Fallback: find any directory with analysis.csv
        for cand in sorted(reflex_dir.glob("**/analysis.csv")):
            if cand.is_file() and not cand.name.startswith("._"):
                p_dir = cand.parent
                strat = p_dir.parent.name
                participants.append((strat, p_dir.name, p_dir))
    return sorted(participants, key=lambda x: x[1])


def load_time_mapping(time_csv_path: Path) -> Tuple[Dict[int, float], int]:
    """Read time.csv and return mapping of 1-indexed Frame -> elapsed wall-clock Seconds."""
    mapping: Dict[int, float] = {}
    max_frame = 0
    with open(time_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_clean = {k.strip(): v for k, v in row.items() if k is not None}
            f_str = f_clean.get("Frame", "")
            s_str = f_clean.get("Seconds", "")
            if f_str and s_str:
                f_idx = int(float(f_str))
                sec = float(s_str)
                mapping[f_idx] = sec
                if f_idx > max_frame:
                    max_frame = f_idx
    return mapping, max_frame


def load_speech_utterances(speech_csv_path: Path) -> List[Dict[str, Any]]:
    """Load utterances from speech.csv."""
    utterances: List[Dict[str, Any]] = []
    if not speech_csv_path.is_file():
        return utterances
    with open(speech_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_clean = {k.strip(): (v.strip() if v is not None else "") for k, v in row.items() if k is not None}
            speaker = f_clean.get("Id", "").lower()
            try:
                b_time = float(f_clean.get("BeginTime", "0"))
                e_time = float(f_clean.get("EndTime", "0"))
            except ValueError:
                continue
            transcript = f_clean.get("Transcript", "")
            utterances.append({
                "speaker": speaker,
                "begin_time": b_time,
                "end_time": e_time,
                "transcript": transcript,
            })
    return utterances


def parse_speech_overlap(
    utterances: List[Dict[str, Any]],
    wall_window_sec: List[float],
) -> Tuple[int, str]:
    """Check participant speech in wall-clock window and return (spoke, transcript)."""
    w_start, w_end = wall_window_sec[0], wall_window_sec[1]
    matching_transcripts: List[str] = []
    spoke = 0

    for u in utterances:
        if u["speaker"] in {"participant", "robot/participant"}:
            if u["begin_time"] <= w_end and u["end_time"] >= w_start:
                spoke = 1
                t = u["transcript"].strip()
                if t and t.upper() != "NA":
                    matching_transcripts.append(t)

    transcript = " ".join(matching_transcripts).strip()
    return spoke, transcript


def build_reflex_items(
    dataset: str,
    raw_dir: Optional[Path] = None,
    check_paths: bool = True,
) -> Tuple[List[Item], Dict[str, Any]]:
    """Build items and stats for reflex or reflex_resolve."""
    if dataset not in ("reflex", "reflex_resolve"):
        raise ValueError(f"Unknown dataset: {dataset}. Must be 'reflex' or 'reflex_resolve'")

    reflex_raw = get_reflex_raw_dir(raw_dir)
    participants = discover_participants(reflex_raw)

    items: List[Item] = []
    stats: Dict[str, Any] = {
        "dataset": dataset,
        "n_participants": len(participants),
        "episodes_seen": 0,
        "items_kept": 0,
        "items_per_class": {"0": 0, "1": 0},
        "skip_counts": {
            "missing_video": 0,
            "invalid_frames": 0,
            "invalid_window": 0,
            "missing_resolved_label": 0,
        },
    }

    for strat, p_id, p_dir in participants:
        video_path = p_dir / "video_cam1.mp4"
        analysis_path = p_dir / "analysis.csv"
        time_path = p_dir / "time.csv"
        speech_path = p_dir / "speech.csv"

        if not video_path.is_file() and check_paths:
            stats["skip_counts"]["missing_video"] += 1
            continue

        try:
            fps, duration, nb_frames = get_video_info(video_path)
        except Exception:
            fps, duration, nb_frames = 40.0 / 9.0, 0.0, 0

        time_map, max_time_frame = load_time_mapping(time_path)
        max_frame = max(nb_frames, max_time_frame)
        utterances = load_speech_utterances(speech_path)

        # Read analysis.csv rows
        rows: List[Dict[str, str]] = []
        with open(analysis_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                row_clean = {}
                for k, v in r.items():
                    if k is not None:
                        key = k.strip()
                        if key == "Resloved?":
                            key = "Resolved?"
                        row_clean[key] = v.strip() if v is not None else ""
                rows.append(row_clean)

        # Episodes are 4-row blocks: Pre -> Failure -> Explanation -> Resolution
        n_rows = len(rows)
        for i in range(0, n_rows - 3, 4):
            block = rows[i : i + 4]
            pre_r, fail_r, exp_r, res_r = block[0], block[1], block[2], block[3]

            if (
                pre_r.get("State") != "Pre"
                or fail_r.get("State") != "Failure"
                or exp_r.get("State") != "Explanation"
                or res_r.get("State") != "Resolution"
            ):
                # Try finding next block if row alignment is non-standard
                continue

            episode_idx = i // 4
            stats["episodes_seen"] += 1

            action = fail_r.get("Action", "").strip().lower()
            explanation_level = exp_r.get("Explanation Level", "").strip().lower()
            round_no = fail_r.get("Round No.", "")

            # Parse frame indices
            try:
                pre_sf, pre_ef = int(float(pre_r["Start Frame"])), int(float(pre_r["End Frame"]))
                fail_sf, fail_ef = int(float(fail_r["Start Frame"])), int(float(fail_r["End Frame"]))
                _exp_sf, exp_ef = int(float(exp_r["Start Frame"])), int(float(exp_r["End Frame"]))
                _res_sf, res_ef = int(float(res_r["Start Frame"])), int(float(res_r["End Frame"]))
            except (ValueError, KeyError):
                stats["skip_counts"]["invalid_frames"] += 1
                continue

            if pre_sf <= 0 or pre_ef <= pre_sf or fail_sf <= 0 or fail_ef <= fail_sf:
                stats["skip_counts"]["invalid_frames"] += 1
                continue

            if dataset == "reflex":
                # Item 1: pre (label = 1)
                pre_act_win = [
                    frame_to_container_seconds(pre_sf, fps),
                    frame_to_container_seconds(pre_ef, fps) if duration == 0 else round(min(duration, frame_to_container_seconds(pre_ef, fps)), 4),
                ]
                pre_wall_s = time_map.get(pre_sf, frame_to_container_seconds(pre_sf, fps))
                pre_wall_e = time_map.get(pre_ef, frame_to_container_seconds(pre_ef, fps))
                pre_wall_act = [round(pre_wall_s, 4), round(pre_wall_e, 4)]

                if pre_act_win[0] >= pre_act_win[1] or pre_wall_act[0] >= pre_wall_act[1]:
                    stats["skip_counts"]["invalid_window"] += 1
                else:
                    spoke_pre, trans_pre = parse_speech_overlap(utterances, pre_wall_act)
                    item_pre = Item(
                        item_id=f"reflex:{p_id}:{episode_idx}:pre",
                        dataset="reflex",
                        group_id=p_id,
                        video_path=str(video_path.resolve()),
                        action_window_sec=pre_act_win,
                        reaction_window_sec=pre_act_win,
                        label=1,
                        label_source="reflex.analysis_state_pre_vs_failure",
                        context_text=f"Task: human-robot collaborative object sorting. Step: {action} object.",
                        official_split=None,
                        meta={
                            "action_frames": [pre_sf, pre_ef],
                            "reaction_frames": [pre_sf, pre_ef],
                            "wall_action_window_sec": pre_wall_act,
                            "wall_reaction_window_sec": pre_wall_act,
                            "spoke": spoke_pre,
                            "transcript": trans_pre,
                            "verb": action,
                            "noun": "object",
                            "episode_idx": episode_idx,
                            "round_no": round_no,
                            "strategy": strat,
                            "participant": p_id,
                        },
                    )
                    items.append(item_pre)
                    stats["items_kept"] += 1
                    stats["items_per_class"]["1"] += 1

                # Item 2: fail (label = 0)
                fail_act_win = [
                    frame_to_container_seconds(fail_sf, fps),
                    frame_to_container_seconds(fail_ef, fps) if duration == 0 else round(min(duration, frame_to_container_seconds(fail_ef, fps)), 4),
                ]
                fail_ref = min(exp_ef, fail_ef + 22, max_frame if max_frame > 0 else fail_ef + 22)
                fail_react_win = [
                    frame_to_container_seconds(fail_sf, fps),
                    frame_to_container_seconds(fail_ref, fps) if duration == 0 else round(min(duration, frame_to_container_seconds(fail_ref, fps)), 4),
                ]
                fail_wall_act = [round(time_map.get(fail_sf, fail_act_win[0]), 4), round(time_map.get(fail_ef, fail_act_win[1]), 4)]
                fail_wall_react = [round(time_map.get(fail_sf, fail_react_win[0]), 4), round(time_map.get(fail_ref, fail_react_win[1]), 4)]

                if fail_act_win[0] >= fail_act_win[1] or fail_react_win[0] >= fail_react_win[1]:
                    stats["skip_counts"]["invalid_window"] += 1
                else:
                    spoke_fail, trans_fail = parse_speech_overlap(utterances, fail_wall_react)
                    item_fail = Item(
                        item_id=f"reflex:{p_id}:{episode_idx}:fail",
                        dataset="reflex",
                        group_id=p_id,
                        video_path=str(video_path.resolve()),
                        action_window_sec=fail_act_win,
                        reaction_window_sec=fail_react_win,
                        label=0,
                        label_source="reflex.analysis_state_pre_vs_failure",
                        context_text=f"Task: human-robot collaborative object sorting. Step: {action} object.",
                        official_split=None,
                        meta={
                            "action_frames": [fail_sf, fail_ef],
                            "reaction_frames": [fail_sf, fail_ref],
                            "wall_action_window_sec": fail_wall_act,
                            "wall_reaction_window_sec": fail_wall_react,
                            "spoke": spoke_fail,
                            "transcript": trans_fail,
                            "verb": action,
                            "noun": "object",
                            "episode_idx": episode_idx,
                            "round_no": round_no,
                            "strategy": strat,
                            "participant": p_id,
                        },
                    )
                    items.append(item_fail)
                    stats["items_kept"] += 1
                    stats["items_per_class"]["0"] += 1

            elif dataset == "reflex_resolve":
                # Check resolved label on Failure row
                resolved_raw = fail_r.get("Resolved?", "").strip().upper()
                if resolved_raw not in ("TRUE", "FALSE"):
                    stats["skip_counts"]["missing_resolved_label"] += 1
                    continue

                label = 1 if resolved_raw == "TRUE" else 0

                act_win = [
                    frame_to_container_seconds(fail_sf, fps),
                    frame_to_container_seconds(exp_ef, fps) if duration == 0 else round(min(duration, frame_to_container_seconds(exp_ef, fps)), 4),
                ]
                react_win = [
                    frame_to_container_seconds(fail_sf, fps),
                    frame_to_container_seconds(res_ef, fps) if duration == 0 else round(min(duration, frame_to_container_seconds(res_ef, fps)), 4),
                ]
                wall_act = [round(time_map.get(fail_sf, act_win[0]), 4), round(time_map.get(exp_ef, act_win[1]), 4)]
                wall_react = [round(time_map.get(fail_sf, react_win[0]), 4), round(time_map.get(res_ef, react_win[1]), 4)]

                if act_win[0] >= act_win[1] or react_win[0] >= react_win[1]:
                    stats["skip_counts"]["invalid_window"] += 1
                    continue

                spoke, trans = parse_speech_overlap(utterances, wall_react)
                item_res = Item(
                    item_id=f"reflex_resolve:{p_id}:{episode_idx}",
                    dataset="reflex_resolve",
                    group_id=p_id,
                    video_path=str(video_path.resolve()),
                    action_window_sec=act_win,
                    reaction_window_sec=react_win,
                    label=label,
                    label_source="reflex.analysis_resolved",
                    context_text=f"Task: human-robot collaborative object sorting. Step: {action} object. Explanation level: {explanation_level}.",
                    official_split=None,
                    meta={
                        "action_frames": [fail_sf, exp_ef],
                        "reaction_frames": [fail_sf, res_ef],
                        "wall_action_window_sec": wall_act,
                        "wall_reaction_window_sec": wall_react,
                        "spoke": spoke,
                        "transcript": trans,
                        "verb": action,
                        "noun": explanation_level,
                        "episode_idx": episode_idx,
                        "round_no": round_no,
                        "strategy": strat,
                        "participant": p_id,
                    },
                )
                items.append(item_res)
                stats["items_kept"] += 1
                stats["items_per_class"][str(label)] += 1

    return items, stats


def build_reflex_splits(
    items: List[Item],
    dataset: str,
    all_participant_ids: Optional[List[str]] = None,
    force: bool = False,
    path_dir: Union[str, Path] = "splits",
) -> Dict[str, Any]:
    """Build canonical 70/30 grouped split partitioned across the shared 55 participants."""
    if all_participant_ids is not None:
        unique_groups = sorted(list(set(all_participant_ids)))
    else:
        unique_groups = sorted(list(set(it.group_id for it in items)))

    rng = np.random.default_rng(0)
    shuffled = list(unique_groups)
    rng.shuffle(shuffled)

    n_train = int(round(0.7 * len(shuffled)))
    train_groups = set(shuffled[:n_train])

    official_split = {
        it.item_id: ("train" if it.group_id in train_groups else "test")
        for it in items
    }

    return make_group_split(
        items,
        dataset=dataset,
        official=official_split,
        force=force,
        path_dir=path_dir,
        source="grouped_70_30_seed0",
        seed=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reflex and reflex_resolve items and splits")
    parser.add_argument("--dataset", choices=["reflex", "reflex_resolve", "all"], default="all")
    parser.add_argument("--force-split", action="store_true", help="Force overwrite splits")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Raw REFLEX directory")
    args = parser.parse_args()

    datasets = ["reflex", "reflex_resolve"] if args.dataset == "all" else [args.dataset]
    raw_dir = args.raw_dir or (DATA_ROOT / "raw" / "reflex")

    # Discover all participants first to ensure shared split partition
    reflex_raw = get_reflex_raw_dir(raw_dir)
    participants = [p[1] for p in discover_participants(reflex_raw)]

    for ds in datasets:
        print(f"Building items for {ds}...")
        items, stats = build_reflex_items(ds, raw_dir=raw_dir)
        print(f"  episodes seen: {stats['episodes_seen']}, items kept: {stats['items_kept']}")
        print(f"  items per class: {stats['items_per_class']}")
        print(f"  skip counts: {stats['skip_counts']}")

        # Validate items
        errors = validate_items(items, check_paths=True)
        if errors:
            raise RuntimeError(f"Item validation failed for {ds} with {len(errors)} errors:\n" + "\n".join(errors[:10]))
        print(f"  validate_items: 0 errors across {len(items)} items")

        # Save items and stats
        out_dir = DATA_ROOT / "items" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        items_path = out_dir / "items.jsonl"
        write_items(items, items_path)
        print(f"  Wrote {items_path}")

        stats_path = out_dir / "stats.json"
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"  Wrote {stats_path}")

        # Build split
        split_res = build_reflex_splits(
            items,
            dataset=ds,
            all_participant_ids=participants,
            force=args.force_split,
        )
        print(f"  Wrote splits/{ds}.json (train={len(split_res['train'])}, test={len(split_res['test'])})")


if __name__ == "__main__":
    main()
