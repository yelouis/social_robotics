from __future__ import annotations

import json
from pathlib import Path
import tempfile


from harness.items import validate_items
from harness.splits import make_group_split
from sources.oops import build_oops_items, derive_compilation_id


def test_derive_compilation_id():
    assert derive_compilation_id("FailFactory - No Pain, No Gain (Workout Fails)12") == "FailFactory - No Pain, No Gain (Workout Fails)"
    assert derive_compilation_id("Winter Fails! (January 2017) _ FailArmy36") == "Winter Fails! (January 2017) _ FailArmy"
    assert derive_compilation_id("NoDigitsHere") == "NoDigitsHere"
    assert derive_compilation_id("12345") == "12345"


def test_build_oops_items_synthetic():
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        anns_dir = tmp_dir / "annotations"
        anns_dir.mkdir()

        # Create dummy video files
        vid_dir = tmp_dir / "videos"
        vid_dir.mkdir()
        (vid_dir / "clip1.mp4").touch()
        (vid_dir / "clip2.mp4").touch()
        (vid_dir / "clip3.mp4").touch()

        # Synthetic annotations
        tt = {
            # 1. Valid train clip
            "Compilation A 1": {
                "t": [5.0, 5.2, 5.1],
                "len": 10.0,
                "stdev": 0.1,
                "n_notfound": 0,
            },
            # 2. Missing onsets (n_notfound = 1)
            "Compilation A 2": {
                "t": [-1.0, 5.0, 5.1],
                "len": 10.0,
                "stdev": 2.5,
                "n_notfound": 1,
            },
            # 3. High stdev (> 1.0)
            "Compilation B 1": {
                "t": [4.0, 5.5, 7.0],
                "len": 12.0,
                "stdev": 1.5,
                "n_notfound": 0,
            },
            # 4. Pre window out of bounds (t - 4.0 < 0)
            "Compilation B 2": {
                "t": [3.0, 3.1, 3.2],
                "len": 10.0,
                "stdev": 0.1,
                "n_notfound": 0,
            },
            # 5. Post window out of bounds (t + 3.0 > len)
            "Compilation C 1": {
                "t": [7.0, 7.1, 7.2],
                "len": 9.0,
                "stdev": 0.1,
                "n_notfound": 0,
            },
            # 6. Valid val clip
            "Compilation C 2": {
                "t": [6.0, 6.1, 6.2],
                "len": 12.0,
                "stdev": 0.1,
                "n_notfound": 0,
            },
        }
        (anns_dir / "transition_times.json").write_text(json.dumps(tt), encoding="utf-8")
        (anns_dir / "train.txt").write_text("Compilation A 1\nCompilation A 2\nCompilation B 1\nCompilation B 2\n", encoding="utf-8")
        (anns_dir / "val.txt").write_text("Compilation C 1\nCompilation C 2\n", encoding="utf-8")

        items, stats, official_map = build_oops_items(
            oops_dir=tmp_dir,
            anns_dir=anns_dir,
            check_paths=False,
        )

        assert stats["clips_seen"] == 6
        assert stats["clips_kept"] == 2  # Compilation A 1 and Compilation C 2
        assert stats["skipped"]["missing_onsets"] == 1
        assert stats["skipped"]["stdev_too_high"] == 1
        assert stats["skipped"]["pre_oob"] == 1
        assert stats["skipped"]["post_oob"] == 1
        assert stats["clips_seen"] == stats["clips_kept"] + sum(stats["skipped"].values())

        assert len(items) == 4  # 2 clips * 2 items each
        assert stats["items_per_split"] == {"train": 2, "test": 2}

        # Check item windows and labels
        pre_item = next(it for it in items if it.item_id == "oops:Compilation A 1:pre")
        post_item = next(it for it in items if it.item_id == "oops:Compilation A 1:post")

        assert pre_item.label == 1
        assert pre_item.action_window_sec == [1.1, 4.1]  # 5.1 - 4.0, 5.1 - 1.0
        assert pre_item.reaction_window_sec == [1.1, 4.1]
        assert pre_item.context_text == "A short clip from a home video."
        assert pre_item.group_id == "Compilation A"
        assert pre_item.dataset == "oops"

        assert post_item.label == 0
        assert post_item.action_window_sec == [5.1, 8.1]  # 5.1, 5.1 + 3.0
        assert post_item.reaction_window_sec == [5.1, 8.1]

        # Leakage check: every item has the constant string
        for it in items:
            assert it.context_text == "A short clip from a home video."

        # Validate items
        errors = validate_items(items, check_paths=False)
        assert errors == []

        # Make group split
        splits_dir = tmp_dir / "splits"
        splits_dir.mkdir()
        split_res = make_group_split(items, "oops", official=official_map, path_dir=splits_dir)
        assert len(split_res["train"]) == 2
        assert len(split_res["test"]) == 2
        assert split_res["source"] == "official"
