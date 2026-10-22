from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config import DATA_ROOT
from harness.items import Item, validate_items


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
    candidates: List[Item] = []

    for rec in records:
        split = rec.get("split", "train")
        if split not in ("train", "test"):
            continue

        video_field = rec.get("video", "")
        task = rec.get("task", "").strip()
        reward = rec.get("reward")
        if reward is None:
            skip_counts["missing_reward"] += 1
            continue

        file_stem = rec.get("file_stem") or Path(video_field).stem
        source = rec.get("source")
        if not source:
            # Try to infer source from path or file_stem
            p_parts = Path(video_field).parts
            if len(p_parts) > 1 and p_parts[0] != split:
                source = p_parts[0]
            elif "robo_arena" in file_stem:
                source = "robo_arena"
            else:
                source = file_stem.split("_")[0]

        label, reason = parse_roboreward_label(source, file_stem, reward)
        if label is None:
            skip_counts[reason or "unhandled_label"] += 1
            continue

        # Duration
        dur_val = rec.get("duration")
        video_p = base_dir / video_field if video_field else (base_dir / split / f"{file_stem}.mp4")

        if check_paths and not video_p.exists():
            skip_counts["missing_video"] += 1
            continue

        if dur_val is None:
            if check_paths and video_p.exists():
                try:
                    dur_val = get_video_duration(video_p)
                except Exception:
                    skip_counts["ffprobe_error"] += 1
                    continue
            else:
                dur_val = 5.0
        duration = float(dur_val)

        group_id = derive_roboreward_group_id(source, file_stem)
        item_id = f"roboreward:{split}:{file_stem}"
        context_text = f"Task: {task}." if not task.endswith(".") else f"Task: {task}"

        meta = {
            "source": source,
            "file_stem": file_stem,
            "raw_reward": reward,
            "task": task,
            "split": split,
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
        "straddling_groups_count": len(straddling),
        "straddling_train_items_removed": len(train_cands) - len(train_pool_eligible),
        "train_pool_eligible": len(train_pool_eligible),
        "train_pool_sampled": len(train_pool),
        "train_pool_success": len(sampled_succ),
        "train_pool_failure": len(sampled_fail),
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
