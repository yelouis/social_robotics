from __future__ import annotations

import csv
from pathlib import Path
import subprocess
from typing import List, Tuple

from harness.items import validate_items
from sources.reflex import (
    build_reflex_items,
    build_reflex_splits,
    frame_to_container_seconds,
)


def create_synthetic_reflex_fixture(tmp_path: Path) -> Tuple[Path, List[str]]:
    """Create a synthetic REFLEX dataset directory with 2 strategies and 4 participants."""
    dataset_dir = tmp_path / "raw" / "reflex" / "Dataset"
    participants = ["C1-1", "C1-2", "D1-1", "D1-2"]
    strategies = {
        "C1-1": "C1-Fixed-Low",
        "C1-2": "C1-Fixed-Low",
        "D1-1": "D1-Decay-Smooth",
        "D1-2": "D1-Decay-Smooth",
    }

    for p in participants:
        strat = strategies[p]
        p_dir = dataset_dir / strat / p
        p_dir.mkdir(parents=True, exist_ok=True)

        # Generate a synthetic 60-second video_cam1.mp4 via ffmpeg
        mp4_path = p_dir / "video_cam1.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=60",
            "-r", "40/9", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(mp4_path),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Write time.csv (Frames 1..200, 1-indexed)
        time_csv = p_dir / "time.csv"
        with open(time_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Stamp", "Timestamp", "Seconds", "Time"])
            for f_idx in range(1, 201):
                sec = round((f_idx - 1) * 0.225, 4)
                writer.writerow([f_idx, 0, 0, sec, f"00:00:{sec:04.1f}"])

        # Write analysis.csv (Contiguous 4-row blocks: Pre, Failure, Explanation, Resolution)
        # Episode 0: Resolved = TRUE, Object has banned failure word "White Cube - Drop"
        # Episode 1: Resolved = FALSE, Object has banned failure word "Cup (Wrong Cup)"
        # Episode 2: Resolved = NA (should be skipped in reflex_resolve as missing_resolved_label)
        analysis_csv = p_dir / "analysis.csv"
        with open(analysis_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Participant", "Strategy", "Round No.", "Object", "Action",
                "Explanation Level", "State", "Start Frame", "End Frame", "Start", "End", "Resolved?"
            ])
            # Episode 0
            writer.writerow([p, strat, 1, "White Cube - Drop", "Pick", "Low", "Pre", 1, 20, 0.0, 4.5, ""])
            writer.writerow([p, strat, 1, "White Cube - Drop", "Pick", "Low", "Failure", 21, 40, 4.5, 9.0, "TRUE"])
            writer.writerow([p, strat, 1, "White Cube - Drop", "Pick", "Low", "Explanation", 41, 60, 9.0, 13.5, ""])
            writer.writerow([p, strat, 1, "White Cube - Drop", "Pick", "Low", "Resolution", 61, 80, 13.5, 18.0, ""])

            # Episode 1
            writer.writerow([p, strat, 2, "Cup (Wrong Cup)", "Place", "High", "Pre", 81, 100, 18.0, 22.5, ""])
            writer.writerow([p, strat, 2, "Cup (Wrong Cup)", "Place", "High", "Failure", 101, 120, 22.5, 27.0, "FALSE"])
            writer.writerow([p, strat, 2, "Cup (Wrong Cup)", "Place", "High", "Explanation", 121, 150, 27.0, 33.75, ""])
            writer.writerow([p, strat, 2, "Cup (Wrong Cup)", "Place", "High", "Resolution", 151, 180, 33.75, 40.5, ""])

            # Episode 2: Missing resolved label
            writer.writerow([p, strat, 3, "Red Toy Fail", "Carry", "Medium", "Pre", 181, 185, 40.5, 41.6, ""])
            writer.writerow([p, strat, 3, "Red Toy Fail", "Carry", "Medium", "Failure", 186, 190, 41.6, 42.7, "NA"])
            writer.writerow([p, strat, 3, "Red Toy Fail", "Carry", "Medium", "Explanation", 191, 195, 42.7, 43.8, ""])
            writer.writerow([p, strat, 3, "Red Toy Fail", "Carry", "Medium", "Resolution", 196, 200, 43.8, 45.0, ""])

        # Write speech.csv
        speech_csv = p_dir / "speech.csv"
        with open(speech_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = ["Id", "BeginTime", "EndTime", "Transcript"] + [f"prosody_{i}" for i in range(48)]
            writer.writerow(headers)
            # Robot utterance during Episode 0 (should be ignored by participant speech filter)
            writer.writerow(["robot", 5.0, 8.0, "I have dropped the object."] + [0.1] * 48)
            # Participant utterance during Episode 0 reaction window
            writer.writerow(["participant", 6.0, 9.5, "Oh what happened"] + [0.2] * 48)
            # Participant utterance during Episode 1 reaction window
            writer.writerow(["robot/participant", 24.0, 26.0, "Okay let us fix this"] + [0.3] * 48)

    return dataset_dir, participants


def test_frame_to_container_seconds():
    fps = 40.0 / 9.0  # 4.4444...
    # Frame 1 -> 0.0 s
    assert frame_to_container_seconds(1, fps) == 0.0
    # Frame 21 -> 20 / (40/9) = 20 * 9 / 40 = 4.5 s
    assert frame_to_container_seconds(21, fps) == 4.5
    # Frame 41 -> 40 * 9 / 40 = 9.0 s
    assert frame_to_container_seconds(41, fps) == 9.0


def test_synthetic_reflex_source_and_windows(tmp_path: Path):
    dataset_dir, participants = create_synthetic_reflex_fixture(tmp_path)

    # Build reflex items (pre vs fail)
    reflex_items, stats_reflex = build_reflex_items("reflex", raw_dir=dataset_dir.parent)
    assert len(reflex_items) == 4 * 3 * 2  # 4 participants * 3 episodes * 2 items (pre and fail)
    assert stats_reflex["episodes_seen"] == 12
    assert stats_reflex["items_kept"] == 24
    assert stats_reflex["items_per_class"] == {"0": 12, "1": 12}

    # Validate items with harness validator (check_paths=True)
    val_errs = validate_items(reflex_items, check_paths=True)
    assert len(val_errs) == 0, f"Validation errors: {val_errs}"

    # Check item details for Episode 0 of C1-1
    pre_item = next(it for it in reflex_items if it.item_id == "reflex:C1-1:0:pre")
    assert pre_item.label == 1
    assert pre_item.label_source == "reflex.analysis_state_pre_vs_failure"
    assert pre_item.action_window_sec == [0.0, frame_to_container_seconds(20, 40.0 / 9.0)]
    assert pre_item.reaction_window_sec == pre_item.action_window_sec
    assert pre_item.meta["action_frames"] == [1, 20]
    assert pre_item.meta["reaction_frames"] == [1, 20]
    assert pre_item.meta["wall_action_window_sec"] == [0.0, round(19 * 0.225, 4)]
    assert pre_item.context_text == "Task: human-robot collaborative object sorting. Step: pick object."
    assert pre_item.meta["verb"] == "pick"
    assert pre_item.meta["noun"] == "object"

    fail_item = next(it for it in reflex_items if it.item_id == "reflex:C1-1:0:fail")
    assert fail_item.label == 0
    assert fail_item.label_source == "reflex.analysis_state_pre_vs_failure"
    assert fail_item.meta["action_frames"] == [21, 40]
    # Failure.End Frame (40) + 22 = 62, Explanation.End Frame = 60 -> min is 60
    assert fail_item.meta["reaction_frames"] == [21, 60]
    assert fail_item.context_text == pre_item.context_text
    assert fail_item.meta["spoke"] == 1
    assert fail_item.meta["transcript"] == "Oh what happened"

    # Build reflex_resolve items
    resolve_items, stats_resolve = build_reflex_items("reflex_resolve", raw_dir=dataset_dir.parent)
    # 4 participants * 2 valid episodes (episodes 0 and 1; episode 2 had NA)
    assert len(resolve_items) == 4 * 2
    assert stats_resolve["episodes_seen"] == 12
    assert stats_resolve["items_kept"] == 8
    assert stats_resolve["items_per_class"] == {"1": 4, "0": 4}
    assert stats_resolve["skip_counts"]["missing_resolved_label"] == 4

    val_errs_res = validate_items(resolve_items, check_paths=True)
    assert len(val_errs_res) == 0, f"Validation errors: {val_errs_res}"

    # Episode 0 of C1-1 (Resolved = TRUE -> label 1)
    res_it_0 = next(it for it in resolve_items if it.item_id == "reflex_resolve:C1-1:0")
    assert res_it_0.label == 1
    assert res_it_0.label_source == "reflex.analysis_resolved"
    assert res_it_0.meta["action_frames"] == [21, 60]  # [Failure.Start, Explanation.End]
    assert res_it_0.meta["reaction_frames"] == [21, 80]  # [Failure.Start, Resolution.End]
    assert res_it_0.context_text == "Task: human-robot collaborative object sorting. Step: pick object. Explanation level: low."
    assert res_it_0.meta["verb"] == "pick"
    assert res_it_0.meta["noun"] == "low"
    assert res_it_0.meta["spoke"] == 1
    assert res_it_0.meta["transcript"] == "Oh what happened"

    # Episode 1 of C1-1 (Resolved = FALSE -> label 0)
    res_it_1 = next(it for it in resolve_items if it.item_id == "reflex_resolve:C1-1:1")
    assert res_it_1.label == 0
    assert res_it_1.meta["action_frames"] == [101, 150]
    assert res_it_1.meta["reaction_frames"] == [101, 180]
    assert res_it_1.context_text == "Task: human-robot collaborative object sorting. Step: place object. Explanation level: high."
    assert res_it_1.meta["verb"] == "place"
    assert res_it_1.meta["noun"] == "high"


def test_no_object_leakage(tmp_path: Path):
    dataset_dir, _ = create_synthetic_reflex_fixture(tmp_path)
    banned_words = {"drop", "wrong", "fail", "true", "false", "resolved"}

    for ds in ("reflex", "reflex_resolve"):
        items, _ = build_reflex_items(ds, raw_dir=dataset_dir.parent)
        for it in items:
            c_lower = it.context_text.lower()
            n_lower = it.meta.get("noun", "").lower()
            for w in banned_words:
                assert w not in c_lower, f"Banned word '{w}' leaked into context_text: {it.context_text}"
                assert w not in n_lower, f"Banned word '{w}' leaked into meta.noun: {it.meta.get('noun')}"


def test_participant_splits_consistency(tmp_path: Path):
    dataset_dir, participants = create_synthetic_reflex_fixture(tmp_path)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    reflex_items, _ = build_reflex_items("reflex", raw_dir=dataset_dir.parent)
    resolve_items, _ = build_reflex_items("reflex_resolve", raw_dir=dataset_dir.parent)

    split_reflex = build_reflex_splits(reflex_items, "reflex", all_participant_ids=participants, path_dir=splits_dir, force=True)
    split_resolve = build_reflex_splits(resolve_items, "reflex_resolve", all_participant_ids=participants, path_dir=splits_dir, force=True)

    # Participant assignment must be identical across both datasets
    tr_groups_reflex = set(it.group_id for it in reflex_items if it.item_id in set(split_reflex["train"]))
    te_groups_reflex = set(it.group_id for it in reflex_items if it.item_id in set(split_reflex["test"]))

    tr_groups_resolve = set(it.group_id for it in resolve_items if it.item_id in set(split_resolve["train"]))
    te_groups_resolve = set(it.group_id for it in resolve_items if it.item_id in set(split_resolve["test"]))

    assert tr_groups_reflex == tr_groups_resolve
    assert te_groups_reflex == te_groups_resolve
    assert len(tr_groups_reflex.intersection(te_groups_reflex)) == 0  # 0 straddling groups
