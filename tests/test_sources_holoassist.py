from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from config import DATA_ROOT
from harness.items import validate_items
from harness.splits import load_split
from sources.holoassist import (
    build_holoassist_items,
    build_holoassist_pool,
    derive_group_id,
    sanitize_task_name,
)


FORBIDDEN_WORDS_RE = re.compile(
    r"\b(mistake|correct|wrong|error|fix|instead)\b",
    re.IGNORECASE,
)


def test_sanitize_task_name():
    assert sanitize_task_name("fix motorcycle") == "repair motorcycle"
    assert sanitize_task_name("setup gopro") == "setup gopro"
    assert sanitize_task_name("assemble stool") == "assemble stool"
    assert not FORBIDDEN_WORDS_RE.search(sanitize_task_name("fix motorcycle"))


def test_derive_group_id():
    assert derive_group_id("R0027-12-GoPro") == "R0027"
    assert derive_group_id("z114-1-printer") == "z114"
    assert derive_group_id("standalone_session") == "standalone_session"


def test_holoassist_builder_synthetic(tmp_path: Path):
    """Fast test of build_holoassist_items using synthetic annotations in tmp_path."""
    synth_sessions = [
        {
            "video_name": "R001-1-GoPro",
            "taskType": "setup gopro",
            "videoMetadata": {"duration": {"seconds": 60.0}},
            "events": [
                {
                    "label": "Conversation",
                    "start": 1.0,
                    "end": 3.0,
                    "attributes": {
                        "Conversation Purpose": "instructor-guidance",
                        "Transcription": "turn it on",
                    },
                },
                {
                    "id": 101,
                    "label": "Fine grained action",
                    "start": 2.0,
                    "end": 4.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "turn",
                        "Noun": "power",
                    },
                },
                {
                    "id": 102,
                    "label": "Fine grained action",
                    "start": 5.0,
                    "end": 7.0,
                    "attributes": {
                        "Action Correctness": "Wrong Action, corrected by student",
                        "Verb": "press",
                        "Noun": "shutter",
                    },
                },
            ],
        },
        {
            "video_name": "R002-1-Switch",
            "taskType": "setup switch",
            "videoMetadata": {"duration": {"seconds": 50.0}},
            "events": [
                {
                    "id": 201,
                    "label": "Fine grained action",
                    "start": 1.0,
                    "end": 3.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "insert",
                        "Noun": "cartridge",
                    },
                },
                {
                    "id": 202,
                    "label": "Fine grained action",
                    "start": 4.0,
                    "end": 6.0,
                    "attributes": {
                        "Action Correctness": "otherwise",
                        "Verb": "close",
                        "Noun": "flap",
                    },
                },
            ],
        },
        {
            "video_name": "R003-1-Coffee",
            "taskType": "make coffee",
            "videoMetadata": {"duration": {"seconds": 40.0}},
            "events": [
                {
                    "id": 301,
                    "label": "Fine grained action",
                    "start": 2.0,
                    "end": 3.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "pour",
                        "Noun": "water",
                    },
                },
                {
                    "id": 302,
                    "label": "Fine grained action",
                    "start": 5.0,
                    "end": 6.0,
                    "attributes": {
                        "Action Correctness": "Wrong Action, not corrected",
                        "Verb": "press",
                        "Noun": "brew",
                    },
                },
            ],
        },
    ]

    ann_file = tmp_path / "data-annotation-trainval-v1_1.json"
    ann_file.write_text(json.dumps(synth_sessions), encoding="utf-8")

    items, stats, split_map = build_holoassist_items(
        labels_dir=tmp_path,
        check_paths=False,
        train_cap=10,
        test_cap=10,
        seed=0,
    )

    errs = validate_items(items, check_paths=False)
    assert not errs, f"Validation errors: {errs}"

    # Verify item_id format starts with holoassist:
    assert len(items) > 0
    assert all(i.item_id.startswith("holoassist:") for i in items)
    assert all(i.meta.get("event_id") is not None for i in items)

    # Check that spoke flag and transcript were captured for first session action
    gopro_items = [i for i in items if "R001" in i.item_id]
    assert any(i.meta.get("spoke") == 1 for i in gopro_items)
    assert any("turn it on" in i.meta.get("transcript", "") for i in gopro_items)

    # Group disjointness
    tr_groups = set(it.group_id for it in items if split_map[it.item_id] == "train")
    te_groups = set(it.group_id for it in items if split_map[it.item_id] == "test")
    assert tr_groups.isdisjoint(te_groups), "Train and test groups must be disjoint!"
    assert stats["straddling_groups"] == 0


def test_holoassist_builder_with_allowlist_synthetic(tmp_path: Path):
    """Test build_holoassist_items with dataset='holoassist_audible' and session_allowlist."""
    synth_sessions = [
        {
            "video_name": "R001-1-GoPro",
            "taskType": "setup gopro",
            "videoMetadata": {"duration": {"seconds": 60.0}},
            "events": [
                {
                    "label": "Conversation",
                    "start": 1.0,
                    "end": 3.0,
                    "attributes": {
                        "Conversation Purpose": "instructor-guidance",
                        "Transcription": "turn it on",
                    },
                },
                {
                    "id": 101,
                    "label": "Fine grained action",
                    "start": 2.0,
                    "end": 4.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "turn",
                        "Noun": "power",
                    },
                },
                {
                    "id": 102,
                    "label": "Fine grained action",
                    "start": 5.0,
                    "end": 7.0,
                    "attributes": {
                        "Action Correctness": "Wrong Action, corrected by student",
                        "Verb": "press",
                        "Noun": "shutter",
                    },
                },
            ],
        },
        {
            "video_name": "R002-1-Switch",
            "taskType": "setup switch",
            "videoMetadata": {"duration": {"seconds": 50.0}},
            "events": [
                {
                    "id": 201,
                    "label": "Fine grained action",
                    "start": 1.0,
                    "end": 3.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "insert",
                        "Noun": "cartridge",
                    },
                },
                {
                    "id": 202,
                    "label": "Fine grained action",
                    "start": 4.0,
                    "end": 6.0,
                    "attributes": {
                        "Action Correctness": "otherwise",
                        "Verb": "close",
                        "Noun": "flap",
                    },
                },
            ],
        },
        {
            "video_name": "R003-1-Coffee",
            "taskType": "make coffee",
            "videoMetadata": {"duration": {"seconds": 40.0}},
            "events": [
                {
                    "id": 301,
                    "label": "Fine grained action",
                    "start": 2.0,
                    "end": 3.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "pour",
                        "Noun": "water",
                    },
                },
                {
                    "id": 302,
                    "label": "Fine grained action",
                    "start": 5.0,
                    "end": 6.0,
                    "attributes": {
                        "Action Correctness": "Wrong Action, not corrected",
                        "Verb": "press",
                        "Noun": "brew",
                    },
                },
            ],
        },
    ]

    ann_file = tmp_path / "data-annotation-trainval-v1_1.json"
    ann_file.write_text(json.dumps(synth_sessions), encoding="utf-8")

    # 1. Build without allowlist (full)
    items_full, stats_full, split_map_full = build_holoassist_items(
        labels_dir=tmp_path,
        check_paths=False,
        train_cap=10,
        test_cap=10,
        seed=0,
    )

    # 2. Build with allowlist (only R001 and R002 allowed, R003 skipped as not_audible)
    allowlist = {"R001-1-GoPro", "R002-1-Switch"}
    items_aud, stats_aud, split_map_aud = build_holoassist_items(
        dataset="holoassist_audible",
        session_allowlist=allowlist,
        labels_dir=tmp_path,
        check_paths=False,
        train_cap=10,
        test_cap=10,
        seed=0,
    )

    errs = validate_items(items_aud, check_paths=False)
    assert not errs, f"Validation errors: {errs}"

    # All items have dataset='holoassist_audible' and prefix
    assert len(items_aud) > 0
    assert all(it.dataset == "holoassist_audible" for it in items_aud)
    assert all(it.item_id.startswith("holoassist_audible:") for it in items_aud)

    # meta.source_item_id is set to holoassist:<vname>:<idx>
    for it in items_aud:
        assert it.meta.get("source_item_id") is not None
        assert it.meta["source_item_id"].startswith("holoassist:")
        assert it.meta["source_item_id"] == it.item_id.replace("holoassist_audible:", "holoassist:")

    # R003 actions were skipped with not_audible
    assert stats_aud["actions_skipped"]["not_audible"] == 2
    assert not any("R003" in it.item_id for it in items_aud)

    # Group assignment is identical for allowed items
    for it in items_aud:
        orig_id = it.meta["source_item_id"]
        assert split_map_aud[it.item_id] == split_map_full[orig_id], (
            f"Group assignment changed for {it.item_id}!"
        )


@pytest.mark.slow
def test_holoassist_builder_and_group_disjointness():
    items, stats, split_map = build_holoassist_items(
        check_paths=False,
        train_cap=50,
        test_cap=30,
        seed=0,
    )

    errs = validate_items(items, check_paths=False)
    assert not errs, f"Validation errors: {errs}"

    assert all(i.item_id.startswith("holoassist:") for i in items)
    assert all(i.meta.get("event_id") is not None for i in items)

    train_items = [it for it in items if split_map[it.item_id] == "train"]
    test_items = [it for it in items if split_map[it.item_id] == "test"]

    assert len(train_items) == 100  # 50 mistake + 50 correct
    assert len(test_items) == 60    # 30 mistake + 30 correct

    assert sum(1 for it in train_items if it.label == 0) == 50
    assert sum(1 for it in train_items if it.label == 1) == 50
    assert sum(1 for it in test_items if it.label == 0) == 30
    assert sum(1 for it in test_items if it.label == 1) == 30

    tr_groups = set(it.group_id for it in train_items)
    te_groups = set(it.group_id for it in test_items)
    assert tr_groups.isdisjoint(te_groups), "Train and test groups must be disjoint!"
    assert stats["straddling_groups"] == 0


@pytest.mark.slow
def test_real_items_leakage_check():
    items_path = DATA_ROOT / "items" / "holoassist" / "items.jsonl"
    if not items_path.exists():
        pytest.skip("HoloAssist items.jsonl not built yet")

    violations = []
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ctx = item.get("context_text", "")
            m = FORBIDDEN_WORDS_RE.search(ctx)
            if m:
                violations.append((item.get("item_id"), ctx, m.group(0)))

    assert not violations, f"Forbidden leakage words found in context_text: {violations[:5]}"


def test_leakage_check_falsification():
    """Demonstrate that the leakage check bites by injecting 'mistake' into a fixture item."""
    clean_item = {
        "item_id": "test_clean",
        "context_text": "Task: setup gopro. Step: approach gopro.",
    }
    assert not FORBIDDEN_WORDS_RE.search(clean_item["context_text"])

    # Injected leak
    leaky_item = {
        "item_id": "test_leaky",
        "context_text": "Task: setup gopro. Step: mistake on gopro.",
    }
    match = FORBIDDEN_WORDS_RE.search(leaky_item["context_text"])
    assert match is not None
    assert match.group(0).lower() == "mistake"


@pytest.mark.slow
def test_real_split_group_disjointness():
    split_path = Path("splits/holoassist.json")
    if not split_path.exists():
        pytest.skip("splits/holoassist.json does not exist yet")

    split_data = load_split("holoassist")
    assert split_data["source"] == "grouped_70_30"
    assert len(split_data["train"]) == 3000
    assert len(split_data["test"]) == 2000

    items_path = DATA_ROOT / "items" / "holoassist" / "items.jsonl"
    if not items_path.exists():
        pytest.skip("items.jsonl does not exist")

    with open(items_path, "r", encoding="utf-8") as f:
        items = {it["item_id"]: it for it in (json.loads(line) for line in f)}

    tr_groups = set(items[iid]["group_id"] for iid in split_data["train"])
    te_groups = set(items[iid]["group_id"] for iid in split_data["test"])

    assert tr_groups.isdisjoint(te_groups), "splits/holoassist.json groups must be strictly disjoint!"


def test_holoassist_pool_builder_synthetic(tmp_path: Path):
    """Test build_holoassist_pool on the 3-session synthetic fixture.

    Requirements:
    - excluded keys never appear;
    - only allowlisted sessions appear;
    - class proportion follows candidates (no balancing);
    - n_total is respected;
    - result is deterministic.
    """
    synth_sessions = [
        {
            "video_name": "R001-1-GoPro",
            "taskType": "setup gopro",
            "videoMetadata": {"duration": {"seconds": 60.0}},
            "events": [
                {
                    "label": "Conversation",
                    "start": 1.0,
                    "end": 3.0,
                    "attributes": {
                        "Conversation Purpose": "instructor-guidance",
                        "Transcription": "turn it on",
                    },
                },
                {
                    "id": 101,
                    "label": "Fine grained action",
                    "start": 2.0,
                    "end": 4.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "turn",
                        "Noun": "power",
                    },
                },
                {
                    "id": 102,
                    "label": "Fine grained action",
                    "start": 5.0,
                    "end": 7.0,
                    "attributes": {
                        "Action Correctness": "Wrong Action, corrected by student",
                        "Verb": "press",
                        "Noun": "shutter",
                    },
                },
            ],
        },
        {
            "video_name": "R002-1-Switch",
            "taskType": "setup switch",
            "videoMetadata": {"duration": {"seconds": 50.0}},
            "events": [
                {
                    "id": 201,
                    "label": "Fine grained action",
                    "start": 1.0,
                    "end": 3.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "insert",
                        "Noun": "cartridge",
                    },
                },
                {
                    "id": 202,
                    "label": "Fine grained action",
                    "start": 4.0,
                    "end": 6.0,
                    "attributes": {
                        "Action Correctness": "otherwise",
                        "Verb": "close",
                        "Noun": "flap",
                    },
                },
            ],
        },
        {
            "video_name": "R003-1-Coffee",
            "taskType": "make coffee",
            "videoMetadata": {"duration": {"seconds": 40.0}},
            "events": [
                {
                    "id": 301,
                    "label": "Fine grained action",
                    "start": 2.0,
                    "end": 3.0,
                    "attributes": {
                        "Action Correctness": "Correct Action",
                        "Verb": "pour",
                        "Noun": "water",
                    },
                },
                {
                    "id": 302,
                    "label": "Fine grained action",
                    "start": 5.0,
                    "end": 6.0,
                    "attributes": {
                        "Action Correctness": "Wrong Action, not corrected",
                        "Verb": "press",
                        "Noun": "brew",
                    },
                },
            ],
        },
    ]

    ann_file = tmp_path / "data-annotation-trainval-v1_1.json"
    ann_file.write_text(json.dumps(synth_sessions), encoding="utf-8")

    allowlist = {"R001-1-GoPro", "R002-1-Switch"}
    # Event index 1 in R001 is the first fine-grained action (eid 101)
    exclude_keys = {("R001-1-GoPro", 1)}

    items, stats = build_holoassist_pool(
        labels_dir=tmp_path,
        check_paths=False,
        session_allowlist=allowlist,
        exclude_keys=exclude_keys,
        n_total=2,
        seed=0,
    )

    errs = validate_items(items, check_paths=False)
    assert not errs, f"Validation errors: {errs}"

    # 1. n_total is respected
    assert len(items) == 2
    assert stats["n_items"] == 2

    # 2. Excluded keys never appear
    item_keys = {(it.meta["session"], it.meta["event_id_idx"]) if "event_id_idx" in it.meta else (it.meta["session"], int(it.item_id.split(":")[-1])) for it in items}
    assert ("R001-1-GoPro", 1) not in item_keys

    # 3. Only allowlisted sessions appear
    for it in items:
        assert it.meta["session"] in allowlist
        assert it.item_id.startswith("holoassist_pool:")
        assert it.meta["source_item_id"].startswith("holoassist:")

    # 4. Result is deterministic
    items2, stats2 = build_holoassist_pool(
        labels_dir=tmp_path,
        check_paths=False,
        session_allowlist=allowlist,
        exclude_keys=exclude_keys,
        n_total=2,
        seed=0,
    )
    assert [it.item_id for it in items] == [it.item_id for it in items2]

    # 5. Class proportion follows candidates (no balancing)
    # When n_total equals total remaining candidates (3), all are taken
    items_all, stats_all = build_holoassist_pool(
        labels_dir=tmp_path,
        check_paths=False,
        session_allowlist=allowlist,
        exclude_keys=exclude_keys,
        n_total=10,  # more than available (3)
        seed=0,
    )
    assert len(items_all) == 3
    # Candidates are: R001:2 (Wrong, label 0), R002:0 (Correct, label 1), R002:1 (Wrong, label 0)
    # Notice: 1 correct and 2 wrong -> natural proportion preserved (not balanced to 1:1)
    labels = [it.label for it in items_all]
    assert labels.count(0) == 2
    assert labels.count(1) == 1

