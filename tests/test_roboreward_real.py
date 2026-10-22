from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from config import DATA_ROOT
from harness.items import read_items
from harness.splits import load_split


@pytest.mark.slow
def test_roboreward_real_items_and_split():
    items_path = DATA_ROOT / "items" / "roboreward" / "items.jsonl"
    stats_path = DATA_ROOT / "items" / "roboreward" / "stats.json"
    splits_path = Path("splits") / "roboreward.json"

    assert items_path.exists(), f"Missing {items_path}"
    assert stats_path.exists(), f"Missing {stats_path}"
    assert splits_path.exists(), f"Missing {splits_path}"

    items = read_items(items_path)
    assert len(items) >= 5000, f"Expected at least 5000 items, got {len(items)}"

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    split_data = load_split("roboreward", path_dir="splits")
    train_ids = set(split_data["train"])
    test_ids = set(split_data["test"])

    # Disjoint splits
    assert not (train_ids & test_ids), "Train and test splits must be disjoint"

    # Train pool size: 2,000 + 2,000 or recorded shortfall
    assert len(train_ids) == stats["train_pool_sampled"]
    assert stats["train_pool_success"] == 2000 or stats.get("train_pool_shortfall_success", 0) > 0
    assert stats["train_pool_failure"] == 2000 or stats.get("train_pool_shortfall_failure", 0) > 0

    # Test size: 1,200 to 2,000 items
    assert 1200 <= len(test_ids) <= 2000, f"Test size {len(test_ids)} outside [1200, 2000]"

    # Check invariants per 04_wave_d_robot_check.md §2.1:
    # 1. No OXE original with reward < 5 and no reward 3 or 4
    for it in items:
        raw_reward = it.meta["raw_reward"]
        source = it.meta["source"]
        stem = it.meta["file_stem"]

        assert raw_reward not in (3, 4, 3.0, 4.0), f"Item {it.item_id} has partial progress reward {raw_reward}"

        is_attempt = "_attempt_" in stem
        if source != "robo_arena" and not is_attempt:
            assert raw_reward == 5.0 or raw_reward == 5, f"OXE original {it.item_id} has reward {raw_reward} < 5"

        # 2. Every _attempt_ item is a failure, and every success is an original or robo_arena
        if is_attempt:
            assert it.label == 0, f"Attempt item {it.item_id} has label {it.label} != 0"

        if it.label == 1:
            assert (not is_attempt) or source == "robo_arena", f"Success item {it.item_id} is an OXE attempt"

    # 3. Groups are correct: for 20 random OXE items, original and attempts share group_id
    oxe_items = [it for it in items if it.meta["source"] != "robo_arena"]
    random.seed(0)
    sample_oxe = random.sample(oxe_items, min(20, len(oxe_items)))
    for it in sample_oxe:
        stem = it.meta["file_stem"]
        source = it.meta["source"]
        idx = stem.find("_attempt_")
        orig_stem = stem[:idx] if idx != -1 else stem
        expected_group = f"{source}:{orig_stem}"
        assert it.group_id == expected_group, f"Group mismatch for {it.item_id}: {it.group_id} != {expected_group}"

    # Check robo_arena cameras share group
    ra_items = [it for it in items if it.meta["source"] == "robo_arena"]
    if ra_items:
        sample_ra = random.sample(ra_items, min(20, len(ra_items)))
        for it in sample_ra:
            stem = it.meta["file_stem"]
            uuid_part = stem.split("_")[0]
            assert it.group_id == f"robo_arena:{uuid_part}"
