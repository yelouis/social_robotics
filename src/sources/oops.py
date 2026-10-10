from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import DATA_ROOT
from harness.items import Item, write_items, validate_items
from harness.splits import make_group_split


def get_oops_dir() -> Path:
    return DATA_ROOT / "raw" / "oops"


def get_annotations_dir(oops_dir: Optional[Path] = None) -> Path:
    base = oops_dir or get_oops_dir()
    for cand in [
        base / "annotations",
        base / "oops_dataset" / "annotations",
        base,
    ]:
        if (cand / "transition_times.json").exists():
            return cand
    return base


def get_video_path(clip_id: str, oops_dir: Optional[Path] = None) -> Path:
    base = oops_dir or get_oops_dir()
    for cand in [
        base / "oops_dataset" / "oops_video" / "train" / f"{clip_id}.mp4",
        base / "oops_dataset" / "oops_video" / "val" / f"{clip_id}.mp4",
        base / "oops_dataset" / "oops_video" / f"{clip_id}.mp4",
        base / "oops_video" / "train" / f"{clip_id}.mp4",
        base / "oops_video" / "val" / f"{clip_id}.mp4",
        base / "oops_video" / f"{clip_id}.mp4",
        base / "oops_dataset" / "video" / f"{clip_id}.mp4",
        base / "oops_dataset" / f"{clip_id}.mp4",
        base / "video" / f"{clip_id}.mp4",
        base / f"{clip_id}.mp4",
    ]:
        if cand.exists():
            return cand
    return base / f"{clip_id}.mp4"


def derive_compilation_id(clip_id: str) -> str:
    """Derive group compilation ID from clip filename by stripping trailing digits."""
    prefix = re.sub(r"\d+$", "", clip_id).strip()
    return prefix if prefix else clip_id


def load_raw_oops_data(
    anns_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], set[str], set[str]]:
    ad = anns_dir or get_annotations_dir()
    tt_path = ad / "transition_times.json"
    if not tt_path.exists():
        raise FileNotFoundError(f"Oops transition_times.json not found in {ad}")
    with open(tt_path, "r", encoding="utf-8") as f:
        tt = json.load(f)

    train_path = ad / "train.txt"
    val_path = ad / "val.txt"
    train_clips: set[str] = set()
    val_clips: set[str] = set()
    if train_path.exists():
        train_clips = set(train_path.read_text(encoding="utf-8").splitlines())
    if val_path.exists():
        val_clips = set(val_path.read_text(encoding="utf-8").splitlines())

    return tt, train_clips, val_clips


def build_oops_items(
    oops_dir: Optional[Path] = None,
    anns_dir: Optional[Path] = None,
    check_paths: bool = True,
    train_cap: int = 2000,
    test_cap: int = 1000,
    seed: int = 0,
) -> Tuple[List[Item], Dict[str, Any], Dict[str, str]]:
    od = oops_dir or get_oops_dir()
    ad = anns_dir or get_annotations_dir(od)
    tt, train_set, val_set = load_raw_oops_data(ad)

    skip_counts: Dict[str, int] = {
        "missing_onsets": 0,
        "stdev_too_high": 0,
        "pre_oob": 0,
        "post_oob": 0,
        "missing_video": 0,
    }

    train_usable: List[Tuple[str, float, float, Path, str]] = []
    val_usable: List[Tuple[str, float, float, Path, str]] = []

    # Sort clips for deterministic processing
    clips_seen = 0
    for clip_id in sorted(tt.keys()):
        clips_seen += 1
        d = tt[clip_id]
        raw_t = d.get("t", [])
        dur = float(d.get("len", 0.0))
        n_notfound = d.get("n_notfound", 0)

        # 1. Missing onsets check
        if n_notfound > 0 or len(raw_t) < 3 or any(x < 0 for x in raw_t):
            skip_counts["missing_onsets"] += 1
            continue

        # 2. Stdev check (> 1.0 s)
        stdev = float(d.get("stdev", 0.0))
        if stdev > 1.0:
            skip_counts["stdev_too_high"] += 1
            continue

        # 3. Median onset
        med_t = float(np.median(raw_t))

        # 4. Pre window out of bounds ([t - 4.0, t - 1.0] -> t - 4.0 < 0)
        if med_t - 4.0 < 0:
            skip_counts["pre_oob"] += 1
            continue

        # 5. Post window out of bounds ([t, t + 3.0] -> t + 3.0 > duration)
        if med_t + 3.0 > dur:
            skip_counts["post_oob"] += 1
            continue

        # 6. Video existence check
        vpath = get_video_path(clip_id, od)
        if check_paths and not vpath.exists():
            skip_counts["missing_video"] += 1
            continue

        # Assign split
        split_name = "train" if clip_id in train_set else ("test" if clip_id in val_set else None)
        if split_name == "train":
            train_usable.append((clip_id, med_t, stdev, vpath, split_name))
        elif split_name == "test":
            val_usable.append((clip_id, med_t, stdev, vpath, split_name))
        else:
            # If not in official train or val, treat as missing split / missing onsets
            skip_counts["missing_onsets"] += 1

    # Apply caps if needed (default_rng(seed))
    rng = np.random.default_rng(seed)
    if len(train_usable) > train_cap:
        indices = rng.choice(len(train_usable), size=train_cap, replace=False)
        indices.sort()
        train_usable = [train_usable[i] for i in indices]

    if len(val_usable) > test_cap:
        indices = rng.choice(len(val_usable), size=test_cap, replace=False)
        indices.sort()
        val_usable = [val_usable[i] for i in indices]

    items: List[Item] = []
    official_map: Dict[str, str] = {}

    all_usable = train_usable + val_usable
    for clip_id, med_t, stdev, vpath, split_name in all_usable:
        comp_id = derive_compilation_id(clip_id)

        # pre item: window [t - 4.0, t - 1.0], label = 1
        pre_id = f"oops:{clip_id}:pre"
        pre_item = Item(
            item_id=pre_id,
            dataset="oops",
            group_id=comp_id,
            video_path=str(vpath.resolve()),
            action_window_sec=[round(med_t - 4.0, 3), round(med_t - 1.0, 3)],
            reaction_window_sec=[round(med_t - 4.0, 3), round(med_t - 1.0, 3)],
            label=1,
            label_source="oops.transition_window",
            context_text="A short clip from a home video.",
            official_split=split_name,
            meta={
                "clip_id": clip_id,
                "window_type": "pre",
                "t": round(med_t, 3),
                "stdev": round(stdev, 3),
                "split": split_name,
            },
        )
        items.append(pre_item)
        official_map[pre_id] = split_name

        # post item: window [t, t + 3.0], label = 0
        post_id = f"oops:{clip_id}:post"
        post_item = Item(
            item_id=post_id,
            dataset="oops",
            group_id=comp_id,
            video_path=str(vpath.resolve()),
            action_window_sec=[round(med_t, 3), round(med_t + 3.0, 3)],
            reaction_window_sec=[round(med_t, 3), round(med_t + 3.0, 3)],
            label=0,
            label_source="oops.transition_window",
            context_text="A short clip from a home video.",
            official_split=split_name,
            meta={
                "clip_id": clip_id,
                "window_type": "post",
                "t": round(med_t, 3),
                "stdev": round(stdev, 3),
                "split": split_name,
            },
        )
        items.append(post_item)
        official_map[post_id] = split_name

    clips_kept = len(train_usable) + len(val_usable)
    stats = {
        "clips_seen": clips_seen,
        "clips_kept": clips_kept,
        "skipped": skip_counts,
        "items_per_split": {
            "train": len(train_usable) * 2,
            "test": len(val_usable) * 2,
        },
    }

    return items, stats, official_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Oops! dataset adapter and item builder")
    parser.add_argument("--oops-dir", type=Path, default=None, help="Root directory for Oops! dataset")
    parser.add_argument("--no-check-paths", action="store_true", help="Skip video path existence check")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: DATA_ROOT/items/oops)")
    args = parser.parse_args()

    oops_dir = args.oops_dir or get_oops_dir()
    check_paths = not args.no_check_paths

    items, stats, official_map = build_oops_items(
        oops_dir=oops_dir,
        check_paths=check_paths,
    )

    out_dir = args.out_dir or (DATA_ROOT / "items" / "oops")
    out_dir.mkdir(parents=True, exist_ok=True)

    items_path = out_dir / "items.jsonl"
    stats_path = out_dir / "stats.json"

    write_items(items, items_path)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Built {len(items)} items across {stats['clips_kept']} clips.")
    print(f"Stats: {json.dumps(stats, indent=2)}")

    errors = validate_items(items, check_paths=check_paths)
    if errors:
        print(f"Validation errors ({len(errors)}):")
        for err in errors[:10]:
            print(f"  {err}")
        raise ValueError(f"Validation failed with {len(errors)} errors")
    print("Validation PASSED (0 errors).")
    split_file = Path("splits") / "oops.json"
    if not split_file.exists():
        make_group_split(items, "oops", official=official_map, path_dir="splits")
        print("Generated splits/oops.json")
    else:
        print("splits/oops.json already exists")


if __name__ == "__main__":
    main()
