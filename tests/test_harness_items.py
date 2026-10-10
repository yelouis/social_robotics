from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from harness.items import Item, read_items, validate_items, write_items
from harness.splits import make_group_split, load_split


def test_validate_items_rejections(tmp_path: Path):
    dummy_video = tmp_path / "video.mp4"
    dummy_video.write_bytes(b"dummy")

    # 1. Valid item
    valid_item = Item(
        item_id="oops:clip1:0",
        dataset="oops",
        group_id="grp1",
        video_path=str(dummy_video),
        action_window_sec=[1.0, 4.0],
        reaction_window_sec=[4.0, 7.0],
        label=1,
        label_source="oops.failure_onset",
        context_text="A short clip from a home video.",
    )
    assert validate_items([valid_item]) == []

    # 2. Duplicate item_id
    item_dup = Item(
        item_id="oops:clip1:0",
        dataset="oops",
        group_id="grp2",
        video_path=str(dummy_video),
        action_window_sec=[2.0, 5.0],
        reaction_window_sec=[5.0, 8.0],
        label=0,
        label_source="oops.failure_onset",
        context_text="A short clip from a home video.",
    )
    errs = validate_items([valid_item, item_dup])
    assert f"duplicate item_id: {valid_item.item_id}" in errs

    # 3. Bad window
    item_bad_window = Item(
        item_id="oops:clip2:0",
        dataset="oops",
        group_id="grp2",
        video_path=str(dummy_video),
        action_window_sec=[5.0, 3.0],  # start >= end
        reaction_window_sec=[-1.0, 2.0],  # start < 0
        label=1,
        label_source="oops.failure_onset",
        context_text="A short clip from a home video.",
    )
    errs = validate_items([item_bad_window])
    assert "bad window action_window_sec [5.0, 3.0] in oops:clip2:0" in errs
    assert "bad window reaction_window_sec [-1.0, 2.0] in oops:clip2:0" in errs

    # 4. Label not 0 or 1
    item_bad_label = Item(
        item_id="oops:clip3:0",
        dataset="oops",
        group_id="grp3",
        video_path=str(dummy_video),
        action_window_sec=[1.0, 2.0],
        reaction_window_sec=[2.0, 3.0],
        label=2,
        label_source="oops.failure_onset",
        context_text="A short clip from a home video.",
    )
    errs = validate_items([item_bad_label])
    assert "label must be 0 or 1 in oops:clip3:0" in errs

    # 5. Empty group_id
    item_empty_group = Item(
        item_id="oops:clip4:0",
        dataset="oops",
        group_id="",
        video_path=str(dummy_video),
        action_window_sec=[1.0, 2.0],
        reaction_window_sec=[2.0, 3.0],
        label=0,
        label_source="oops.failure_onset",
        context_text="A short clip from a home video.",
    )
    errs = validate_items([item_empty_group])
    assert "empty group_id in oops:clip4:0" in errs

    # 6. Missing video_path
    missing_path = str(tmp_path / "nonexistent.mp4")
    item_missing_video = Item(
        item_id="oops:clip5:0",
        dataset="oops",
        group_id="grp5",
        video_path=missing_path,
        action_window_sec=[1.0, 2.0],
        reaction_window_sec=[2.0, 3.0],
        label=0,
        label_source="oops.failure_onset",
        context_text="A short clip from a home video.",
    )
    errs = validate_items([item_missing_video], check_paths=True)
    assert f"missing video_path for oops:clip5:0: {missing_path}" in errs


def test_items_read_write_roundtrip(tmp_path: Path):
    items_path = tmp_path / "items.jsonl"
    it1 = Item("ds:1:0", "ds", "g1", "/p1", [1.0, 2.0], [2.0, 3.0], 1, "src", "ctx")
    it2 = Item("ds:2:0", "ds", "g2", "/p2", [1.0, 2.0], [2.0, 3.0], 0, "src", "ctx")
    write_items([it2, it1], items_path)  # write out of order

    loaded = read_items(items_path)
    assert len(loaded) == 2
    assert loaded[0].item_id == "ds:1:0"  # sorted by item_id
    assert loaded[1].item_id == "ds:2:0"


def test_group_disjointness_500_items_37_groups(tmp_path: Path):
    rng = np.random.default_rng(0)
    groups = [f"g_{i}" for i in range(37)]
    items = []
    for i in range(500):
        grp = rng.choice(groups)
        items.append(Item(f"test:{i}:0", "test", grp, "/v", [0.0, 1.0], [1.0, 2.0], 1, "s", "c"))

    split = make_group_split(items, "test", path_dir=tmp_path)
    train_set = set(split["train"])
    test_set = set(split["test"])

    # Disjoint items
    assert train_set.isdisjoint(test_set)
    assert len(train_set) + len(test_set) == 500

    # Group disjointness
    train_groups = set(item.group_id for item in items if item.item_id in train_set)
    test_groups = set(item.group_id for item in items if item.item_id in test_set)
    assert train_groups.isdisjoint(test_groups)


def test_two_runs_produce_byte_identical_split_files(tmp_path: Path):
    items = [
        Item(f"d:{i}:0", "d", f"g_{i % 10}", "/v", [0.0, 1.0], [1.0, 2.0], 1, "s", "c")
        for i in range(50)
    ]
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"

    make_group_split(items, "d", path_dir=dir1)
    make_group_split(items, "d", path_dir=dir2)

    bytes1 = (dir1 / "d.json").read_bytes()
    bytes2 = (dir2 / "d.json").read_bytes()
    assert bytes1 == bytes2


def test_overwrite_refusal(tmp_path: Path):
    items = [Item("d:1:0", "d", "g1", "/v", [0.0, 1.0], [1.0, 2.0], 1, "s", "c")]
    make_group_split(items, "d", path_dir=tmp_path)

    with pytest.raises(FileExistsError) as exc_info:
        make_group_split(items, "d", path_dir=tmp_path, force=False)
    assert str(exc_info.value) == "split exists: splits/d.json (use force=True and say why in the commit)"

    # With force=True, it succeeds
    make_group_split(items, "d", path_dir=tmp_path, force=True)


def test_official_split_straddle_error(tmp_path: Path):
    items = [
        Item("d:1:0", "d", "g1", "/v", [0.0, 1.0], [1.0, 2.0], 1, "s", "c"),
        Item("d:2:0", "d", "g1", "/v", [0.0, 1.0], [1.0, 2.0], 0, "s", "c"),
        Item("d:3:0", "d", "g2", "/v", [0.0, 1.0], [1.0, 2.0], 1, "s", "c"),
    ]
    # g1 has items in both train and test
    official = {
        "d:1:0": "train",
        "d:2:0": "test",
        "d:3:0": "train",
    }
    with pytest.raises(ValueError) as exc_info:
        make_group_split(items, "d", official=official, path_dir=tmp_path)
    assert str(exc_info.value) == "official split is not group-disjoint: 1 groups straddle"


def test_tamper_detection(tmp_path: Path):
    items = [
        Item("d:1:0", "d", "g1", "/v", [0.0, 1.0], [1.0, 2.0], 1, "s", "c"),
        Item("d:2:0", "d", "g2", "/v", [0.0, 1.0], [1.0, 2.0], 0, "s", "c"),
    ]
    make_group_split(items, "d", path_dir=tmp_path)
    split_file = tmp_path / "d.json"

    # Verify normal loading succeeds
    loaded = load_split("d", path_dir=tmp_path)
    assert loaded["dataset"] == "d"

    # Tamper with the file
    with open(split_file, "r") as f:
        data = json.load(f)
    data["train"].append("tampered:99:0")
    with open(split_file, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError) as exc_info:
        load_split("d", path_dir=tmp_path)
    assert "tamper detected: sha256 mismatch for split d" in str(exc_info.value)
