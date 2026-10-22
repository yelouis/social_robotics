from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config import DATA_ROOT
from harness.items import Item, validate_items, write_items
from harness.splits import make_group_split


def get_roboreward_dir() -> Path:
    return DATA_ROOT / "raw" / "roboreward"


def derive_roboreward_group_id(source: str, file_stem: str) -> str:
    """Derive group_id per 04_wave_d_robot_check.md §2.1.

    OXE: <source>:<original episode stem> (suffix _attempt_<k>_score_<s> removed).
    robo_arena: robo_arena:<rollout uuid> (stem before first _).
    """
    if source == "robo_arena" or file_stem.startswith("robo_arena"):
        stem = file_stem
        if stem.startswith("robo_arena:"):
            stem = stem[len("robo_arena:"):]
        uuid_part = stem.split("_")[0]
        return f"robo_arena:{uuid_part}"
    else:
        idx = file_stem.find("_attempt_")
        orig_stem = file_stem[:idx] if idx != -1 else file_stem
        return f"{source}:{orig_stem}"


def parse_roboreward_label(
    source: str,
    file_stem: str,
    reward: float,
) -> Tuple[Optional[int], Optional[str]]:
    """Determine label per vision-only rule (04_wave_d_robot_check.md §2.1).

    Success (1):
      - OXE original (no _attempt_) with reward = 5
      - robo_arena with reward = 5
    Failure (0):
      - OXE attempt (_attempt_) with reward <= 2
      - robo_arena with reward <= 2
    Excluded:
      - reward 3 or 4: partial progress
      - OXE original with reward < 5: counterfactual instruction relabel
      - OXE attempt with reward > 2: attempt not failure
    """
    r = float(reward)
    if r in (3.0, 4.0):
        return None, "partial_progress_3_or_4"

    is_attempt = "_attempt_" in file_stem

    if source == "robo_arena":
        if r == 5.0:
            return 1, None
        elif r <= 2.0:
            return 0, None
        else:
            return None, "partial_progress_3_or_4"
    else:
        if not is_attempt:
            if r == 5.0:
                return 1, None
            else:
                return None, "oxe_original_counterfactual"
        else:
            if r <= 2.0:
                return 0, None
            else:
                return None, "oxe_attempt_reward_not_failure"


def get_video_duration(video_path: Union[str, Path]) -> float:
    """Query video duration via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(res.stdout.strip())


def build_roboreward_items(
    records: Optional[List[Dict[str, Any]]] = None,
    raw_dir: Optional[Path] = None,
    check_paths: bool = True,
    train_pool_cap_per_class: int = 2000,
    seed: int = 0,
) -> Tuple[List[Item], Dict[str, Any], Dict[str, str]]:
    """Build items, stats, and split map for RoboReward."""
    base_dir = raw_dir or get_roboreward_dir()

    if records is None:
        records = []
        for split in ("train", "test"):
            meta_path = base_dir / split / "metadata.jsonl"
            if not meta_path.exists():
                meta_path = base_dir / f"{split}_metadata.jsonl"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            d["split"] = split
                            records.append(d)

    skip_counts: Dict[str, int] = defaultdict(int)
    source_outcome_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    candidates: List[Item] = []

    for rec in records:
        split = rec.get("split", "train")
        if split not in ("train", "test"):
            continue

        video_field = rec.get("file_name") or rec.get("video", "")
        task = rec.get("task", "").strip()
        reward = rec.get("reward")
        if reward is None:
            skip_counts["missing_reward"] += 1
            continue

        file_stem = rec.get("file_stem") or Path(video_field).stem
        source = rec.get("source")
        if not source:
            p_parts = Path(video_field).parts
            if len(p_parts) > 1:
                source = p_parts[1] if p_parts[0] == split else p_parts[0]
            elif "robo_arena" in file_stem:
                source = "robo_arena"
            else:
                source = file_stem.split("_")[0]

        label, reason = parse_roboreward_label(source, file_stem, reward)
        if label is None:
            skip_counts[reason or "unhandled_label"] += 1
            source_outcome_counts[source][f"excluded_{reason or 'unhandled'}"] += 1
            continue

        outcome = "success" if label == 1 else "failure"
        source_outcome_counts[source][outcome] += 1

        # Resolve video path on disk
        if (base_dir / split / video_field).exists():
            video_p = base_dir / split / video_field
        elif (base_dir / video_field).exists():
            video_p = base_dir / video_field
        elif video_field:
            if Path(video_field).parts and Path(video_field).parts[0] == split:
                video_p = base_dir / video_field
            else:
                video_p = base_dir / split / video_field
        else:
            video_p = base_dir / split / f"{file_stem}.mp4"

        if check_paths and not video_p.exists():
            skip_counts["missing_video"] += 1
            continue

        dur_val = rec.get("duration")
        duration = float(dur_val) if dur_val is not None else 0.0

        group_id = derive_roboreward_group_id(source, file_stem)
        item_id = f"roboreward:{split}:{file_stem}"
        context_text = f"Task: {task}." if not task.endswith(".") else f"Task: {task}"

        meta = {
            "source": source,
            "file_stem": file_stem,
            "raw_reward": reward,
            "task": task,
            "split": split,
            "_needs_duration": (dur_val is None),
        }

        item = Item(
            item_id=item_id,
            dataset="roboreward",
            group_id=group_id,
            video_path=str(video_p.resolve()) if check_paths and video_p.exists() else str(video_p),
            action_window_sec=[0.0, round(duration, 3)],
            reaction_window_sec=[0.0, round(duration, 3)],
            label=label,
            label_source="roboreward.reward",
            context_text=context_text,
            meta=meta,
        )
        candidates.append(item)

    # Sort candidates deterministically
    candidates.sort(key=lambda x: x.item_id)

    train_cands = [c for c in candidates if c.meta["split"] == "train"]
    test_cands = [c for c in candidates if c.meta["split"] == "test"]

    train_groups = set(c.group_id for c in train_cands)
    test_groups = set(c.group_id for c in test_cands)
    straddling = train_groups.intersection(test_groups)

    # If any group straddles the two splits, remove that group's train-side items from robot train pool
    train_pool_eligible = [c for c in train_cands if c.group_id not in straddling]

    # Sample train pool: 2,000 successes and 2,000 failures with default_rng(0)
    rng = np.random.default_rng(seed)
    train_success = [c for c in train_pool_eligible if c.label == 1]
    train_failure = [c for c in train_pool_eligible if c.label == 0]

    n_succ = min(len(train_success), train_pool_cap_per_class)
    n_fail = min(len(train_failure), train_pool_cap_per_class)

    if len(train_success) > n_succ:
        idx_s = rng.choice(len(train_success), size=n_succ, replace=False)
        idx_s.sort()
        sampled_succ = [train_success[i] for i in idx_s]
    else:
        sampled_succ = train_success

    if len(train_failure) > n_fail:
        idx_f = rng.choice(len(train_failure), size=n_fail, replace=False)
        idx_f.sort()
        sampled_fail = [train_failure[i] for i in idx_f]
    else:
        sampled_fail = train_failure

    train_pool = sampled_succ + sampled_fail
    train_pool.sort(key=lambda x: x.item_id)
    train_pool_ids = [it.item_id for it in train_pool]

    # Test set is all filtered test items
    test_items = list(test_cands)

    all_items = train_pool + test_items
    all_items.sort(key=lambda x: x.item_id)

    # Probe duration concurrently for items that need it
    items_needing_dur = [it for it in all_items if it.meta.pop("_needs_duration", False)]
    if items_needing_dur:
        if check_paths:
            def _probe(it: Item) -> Tuple[str, float]:
                try:
                    d = get_video_duration(it.video_path)
                except Exception:
                    d = 5.0
                return it.item_id, d

            with ThreadPoolExecutor(max_workers=16) as pool:
                dur_results = dict(pool.map(_probe, items_needing_dur))

            for it in items_needing_dur:
                d = dur_results.get(it.item_id, 5.0)
                it.action_window_sec = [0.0, round(d, 3)]
                it.reaction_window_sec = [0.0, round(d, 3)]
        else:
            for it in items_needing_dur:
                it.action_window_sec = [0.0, 5.0]
                it.reaction_window_sec = [0.0, 5.0]

    errors = validate_items(all_items, check_paths=check_paths)
    if errors:
        raise ValueError(f"RoboReward item validation failed: {errors[:5]}")

    split_map = {
        it.item_id: ("train" if it.meta["split"] == "train" else "test")
        for it in all_items
    }

    stats = {
        "dataset": "roboreward",
        "candidates_total": len(candidates),
        "train_candidates": len(train_cands),
        "test_candidates": len(test_cands),
        "counts_by_source_outcome_reason": {src: dict(cnts) for src, cnts in sorted(source_outcome_counts.items())},
        "straddling_groups_count": len(straddling),
        "straddling_train_items_removed": len(train_cands) - len(train_pool_eligible),
        "train_pool_eligible": len(train_pool_eligible),
        "train_pool_sampled": len(train_pool),
        "train_pool_success": len(sampled_succ),
        "train_pool_failure": len(sampled_fail),
        "train_pool_shortfall_success": max(0, train_pool_cap_per_class - len(sampled_succ)),
        "train_pool_shortfall_failure": max(0, train_pool_cap_per_class - len(sampled_fail)),
        "test_items": len(test_items),
        "test_success": sum(1 for it in test_items if it.label == 1),
        "test_failure": sum(1 for it in test_items if it.label == 0),
        "train_pool_ids": train_pool_ids,
        "actions_skipped": dict(skip_counts),
        "seed": seed,
    }
    # Direct access keys for test assertions
    stats["excluded_partial_progress"] = skip_counts.get("partial_progress_3_or_4", 0)

    return all_items, stats, split_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RoboReward items and splits.")
    parser.add_argument("--build", action="store_true", help="Build items.jsonl, stats.json, and splits/roboreward.json")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train pool sampling")
    parser.add_argument("--force-split", action="store_true", help="Force overwrite splits/roboreward.json if exists")
    parser.add_argument("--no-check-paths", action="store_true", help="Disable checking video path existence")
    args = parser.parse_args()

    if args.build:
        print("Building RoboReward items...")
        items, stats, split_map = build_roboreward_items(
            check_paths=not args.no_check_paths,
            seed=args.seed,
        )
        out_dir = DATA_ROOT / "items" / "roboreward"
        out_dir.mkdir(parents=True, exist_ok=True)
        items_path = out_dir / "items.jsonl"
        stats_path = out_dir / "stats.json"

        write_items(items, items_path)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote {items_path} ({len(items)} items) and {stats_path}")

        print("Generating splits/roboreward.json...")
        split_data = make_group_split(
            items,
            "roboreward",
            official=split_map,
            source="roboreward_official_train_test",
            seed=args.seed,
            force=args.force_split,
        )
        print(f"Split generated: {len(split_data['train'])} train, {len(split_data['test'])} test. sha256={split_data['sha256']}")


if __name__ == "__main__":
    main()
