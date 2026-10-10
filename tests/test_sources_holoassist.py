from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from config import DATA_ROOT
from harness.items import validate_items
from harness.splits import load_split
from sources.holoassist import build_holoassist_items, derive_group_id, sanitize_task_name


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


def test_holoassist_builder_and_group_disjointness():
    items, stats, split_map = build_holoassist_items(
        check_paths=False,
        train_cap=50,
        test_cap=30,
        seed=0,
    )

    errs = validate_items(items, check_paths=False)
    assert not errs, f"Validation errors: {errs}"

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
