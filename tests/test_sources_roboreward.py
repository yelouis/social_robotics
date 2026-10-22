from __future__ import annotations

from pathlib import Path

from sources.roboreward import (
    derive_roboreward_group_id,
    parse_roboreward_label,
    build_roboreward_items,
)


def test_derive_group_id_oxe():
    # Original episode
    assert derive_roboreward_group_id("bridge", "episode_001") == "bridge:episode_001"
    # Attempt episode
    assert derive_roboreward_group_id("bridge", "episode_001_attempt_1_score_2") == "bridge:episode_001"
    assert derive_roboreward_group_id("fractal", "run_abc_attempt_3_score_0") == "fractal:run_abc"


def test_derive_group_id_robo_arena():
    # Rollout UUID is before first underscore
    assert derive_roboreward_group_id("robo_arena", "123e4567-e89b_front_cam") == "robo_arena:123e4567-e89b"
    assert derive_roboreward_group_id("robo_arena", "123e4567-e89b_wrist_cam") == "robo_arena:123e4567-e89b"


def test_parse_roboreward_label_oxe():
    # OXE original with reward 5 -> 1
    assert parse_roboreward_label("bridge", "episode_001", 5.0) == (1, None)
    # OXE original with reward < 5 -> excluded (counterfactual)
    lbl, reason = parse_roboreward_label("bridge", "episode_001", 2.0)
    assert lbl is None
    assert reason == "oxe_original_counterfactual"

    # OXE attempt with reward <= 2 -> 0
    assert parse_roboreward_label("bridge", "episode_001_attempt_1_score_2", 2.0) == (0, None)
    assert parse_roboreward_label("bridge", "episode_001_attempt_1_score_0", 0.0) == (0, None)

    # OXE attempt with reward 5 -> excluded
    lbl, reason = parse_roboreward_label("bridge", "episode_001_attempt_1_score_5", 5.0)
    assert lbl is None
    assert reason == "oxe_attempt_reward_not_failure"

    # Partial progress reward 3 or 4 -> excluded
    lbl, reason = parse_roboreward_label("bridge", "episode_001", 3.0)
    assert lbl is None
    assert reason == "partial_progress_3_or_4"

    lbl, reason = parse_roboreward_label("bridge", "episode_001_attempt_1", 4.0)
    assert lbl is None
    assert reason == "partial_progress_3_or_4"


def test_parse_roboreward_label_robo_arena():
    # robo_arena: reward 5 -> 1, reward <= 2 -> 0
    assert parse_roboreward_label("robo_arena", "uuid_cam1", 5.0) == (1, None)
    assert parse_roboreward_label("robo_arena", "uuid_cam1", 1.0) == (0, None)
    assert parse_roboreward_label("robo_arena", "uuid_cam1", 2.0) == (0, None)

    # reward 3, 4 -> excluded
    lbl, reason = parse_roboreward_label("robo_arena", "uuid_cam1", 3.0)
    assert lbl is None
    assert reason == "partial_progress_3_or_4"


def test_build_roboreward_items_synthetic(tmp_path: Path):
    """Test building synthetic RoboReward items and train pool sampling."""
    records = [
        # Train: OXE original (success) + attempt (failure) -> same group
        {"split": "train", "source": "bridge", "file_stem": "ep1", "reward": 5.0, "task": "pick cup", "duration": 4.5},
        {"split": "train", "source": "bridge", "file_stem": "ep1_attempt_1_score_1", "reward": 1.0, "task": "pick cup", "duration": 3.0},
        # Train: robo_arena success + failure
        {"split": "train", "source": "robo_arena", "file_stem": "uuidA_cam0", "reward": 5.0, "task": "push block", "duration": 5.0},
        {"split": "train", "source": "robo_arena", "file_stem": "uuidB_cam0", "reward": 0.0, "task": "push block", "duration": 5.0},
        # Train: excluded partial progress
        {"split": "train", "source": "robo_arena", "file_stem": "uuidC_cam0", "reward": 3.0, "task": "push block", "duration": 5.0},
        # Test: OXE original (success) + attempt (failure)
        {"split": "test", "source": "bridge", "file_stem": "ep2", "reward": 5.0, "task": "pick cup", "duration": 4.0},
        {"split": "test", "source": "bridge", "file_stem": "ep2_attempt_1_score_2", "reward": 2.0, "task": "pick cup", "duration": 2.5},
    ]

    items, stats, split_map = build_roboreward_items(
        records=records,
        check_paths=False,
        train_pool_cap_per_class=10,
        seed=0,
    )

    # Total kept = 4 train + 2 test = 6 items (uuidC excluded)
    assert len(items) == 6
    assert stats["excluded_partial_progress"] == 1

    train_pool_ids = stats["train_pool_ids"]
    assert len(train_pool_ids) == 4

    # Check context text format: Task: <task>.
    for it in items:
        assert it.context_text.startswith("Task: ")
        assert it.context_text.endswith(".")
        assert it.action_window_sec == [0.0, it.reaction_window_sec[1]]
