from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import DATA_ROOT
from harness.items import Item, write_items, validate_items
from harness.splits import make_group_split


def get_holoassist_dir() -> Path:
    return DATA_ROOT / "raw" / "holoassist"


def get_labels_dir(holoassist_dir: Optional[Path] = None) -> Path:
    base = holoassist_dir or get_holoassist_dir()
    return base / "labels"


def get_videos_dir(holoassist_dir: Optional[Path] = None) -> Path:
    base = holoassist_dir or get_holoassist_dir()
    return base / "videos"


def get_holoassist_video_path(video_name: str, holoassist_dir: Optional[Path] = None) -> Path:
    base = holoassist_dir or get_holoassist_dir()
    candidates = [
        base / "videos" / video_name / "Export_py" / "Video_pitchshift.mp4",
        base / "videos" / video_name / "Video_pitchshift.mp4",
        base / video_name / "Export_py" / "Video_pitchshift.mp4",
        base / video_name / "Video_pitchshift.mp4",
        base / "videos" / f"{video_name}.mp4",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return base / "videos" / video_name / "Export_py" / "Video_pitchshift.mp4"


def derive_group_id(video_name: str) -> str:
    """Derives performer / session group ID from video name prefix."""
    return video_name.split("-")[0] if "-" in video_name else video_name


def sanitize_task_name(task_name: str) -> str:
    """Ensures task name has no forbidden outcome/leakage tokens (e.g. 'fix')."""
    # HoloAssist task 'fix motorcycle' -> 'repair motorcycle'
    clean = re.sub(r"\bfix\b", "repair", task_name, flags=re.IGNORECASE)
    return clean.strip()


def load_raw_annotations(labels_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    ld = labels_dir or get_labels_dir()
    json_path = ld / "data-annotation-trainval-v1_1.json"
    if not json_path.exists():
        raise FileNotFoundError(f"HoloAssist annotations not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_official_splits(labels_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    ld = labels_dir or get_labels_dir()
    splits: Dict[str, List[str]] = {}
    for s_name in ("train", "val", "test"):
        p = ld / f"{s_name}-v1_2.txt"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                splits[s_name] = [line.strip() for line in f if line.strip()]
        else:
            splits[s_name] = []
    return splits


def compute_stats(
    sessions: List[Dict[str, Any]],
    splits: Dict[str, List[str]],
) -> Dict[str, Any]:
    n_sessions = len(sessions)
    n_correct = 0
    n_mistake_wrong = 0
    n_mistake_all = 0
    n_instructor_utterances = 0

    n_correct_spoke = 0
    n_mistake_wrong_spoke = 0
    n_mistake_all_spoke = 0

    for sess in sessions:
        events = sess.get("events", [])
        instructor_spans: List[Tuple[float, float]] = []
        for ev in events:
            if ev.get("label") == "Conversation":
                attrs = ev.get("attributes", {})
                cp = attrs.get("Conversation Purpose", "")
                if cp.startswith("instructor"):
                    n_instructor_utterances += 1
                    u_s = float(ev.get("start", 0.0))
                    u_e = float(ev.get("end", 0.0))
                    instructor_spans.append((u_s, u_e))

        for ev in events:
            if ev.get("label") == "Fine grained action":
                attrs = ev.get("attributes", {})
                ac = attrs.get("Action Correctness")
                if not ac:
                    continue
                a_s = float(ev.get("start", 0.0))
                a_e = float(ev.get("end", 0.0))
                react_s = a_s
                react_e = a_e + 5.0

                spoke = any(max(react_s, u_s) < min(react_e, u_e) for u_s, u_e in instructor_spans)

                if ac == "Correct Action":
                    n_correct += 1
                    if spoke:
                        n_correct_spoke += 1
                elif ac.startswith("Wrong Action"):
                    n_mistake_wrong += 1
                    n_mistake_all += 1
                    if spoke:
                        n_mistake_wrong_spoke += 1
                        n_mistake_all_spoke += 1
                elif ac == "otherwise":
                    n_mistake_all += 1
                    if spoke:
                        n_mistake_all_spoke += 1

    # Official split participant-disjointness check
    prefix_splits: Dict[str, set] = defaultdict(set)
    for s_name, sess_list in splits.items():
        for s_id in sess_list:
            prefix = derive_group_id(s_id)
            prefix_splits[prefix].add(s_name)

    straddling_performers = {p: s for p, s in prefix_splits.items() if len(s) > 1}

    return {
        "n_sessions": n_sessions,
        "n_actions_attribute": n_correct + n_mistake_wrong,
        "n_actions_all_attribute": n_correct + n_mistake_all,
        "n_correct": n_correct,
        "n_mistakes": n_mistake_wrong,
        "mistake_pct": (n_mistake_wrong / (n_correct + n_mistake_wrong) * 100.0) if (n_correct + n_mistake_wrong) else 0.0,
        "n_mistakes_all": n_mistake_all,
        "mistake_all_pct": (n_mistake_all / (n_correct + n_mistake_all) * 100.0) if (n_correct + n_mistake_all) else 0.0,
        "n_instructor_utterances": n_instructor_utterances,
        "n_unique_performer_prefixes": len(prefix_splits),
        "n_straddling_performers": len(straddling_performers),
        "react_spoke_mistake_num": n_mistake_wrong_spoke,
        "react_spoke_mistake_denom": n_mistake_wrong,
        "react_spoke_mistake_rate": (n_mistake_wrong_spoke / n_mistake_wrong) if n_mistake_wrong else 0.0,
        "react_spoke_correct_num": n_correct_spoke,
        "react_spoke_correct_denom": n_correct,
        "react_spoke_correct_rate": (n_correct_spoke / n_correct) if n_correct else 0.0,
    }


def print_stats(stats: Dict[str, Any]) -> None:
    print("HoloAssist Dataset Statistics (--stats):")
    print(f"  Sessions: {stats['n_sessions']}")
    print(f"  Fine-grained actions with mistake/correct attribute: {stats['n_actions_attribute']}")
    print(f"    - Correct actions: {stats['n_correct']}")
    print(f"    - Mistake actions: {stats['n_mistakes']} ({stats['mistake_pct']:.2f}%)")
    print(f"    - Mistake actions (incl. otherwise): {stats['n_mistakes_all']} ({stats['mistake_all_pct']:.2f}%)")
    print(f"  Instructor utterances: {stats['n_instructor_utterances']}")
    print("  Performer IDs in official splits:")
    print(f"    - Unique performer prefixes: {stats['n_unique_performer_prefixes']}")
    print(f"    - Performer IDs in >1 official split: {stats['n_straddling_performers']} (0 means participant-disjoint)")
    print("  react-spoke signal in reaction window [start, end + 5.0]:")
    print(f"    - Mistake actions with instructor utterance: {stats['react_spoke_mistake_num']} / {stats['react_spoke_mistake_denom']} ({stats['react_spoke_mistake_rate']*100:.2f}%)")
    print(f"    - Correct actions with instructor utterance: {stats['react_spoke_correct_num']} / {stats['react_spoke_correct_denom']} ({stats['react_spoke_correct_rate']*100:.2f}%)")


def build_holoassist_items(
    holoassist_dir: Optional[Path] = None,
    labels_dir: Optional[Path] = None,
    check_paths: bool = True,
    train_cap: int = 1500,
    test_cap: int = 1000,
    seed: int = 0,
) -> Tuple[List[Item], Dict[str, Any]]:
    hd = holoassist_dir or get_holoassist_dir()
    ld = labels_dir or get_labels_dir(hd)
    sessions = load_raw_annotations(ld)
    sessions = sorted(sessions, key=lambda s: s.get("video_name", ""))

    skip_counts: Dict[str, int] = {
        "missing_correctness_attr": 0,
        "invalid_action_window": 0,
        "invalid_reaction_window": 0,
        "missing_video": 0,
    }

    # 1. Collect all valid candidates
    candidates: List[Item] = []
    actions_seen = 0

    for sess in sessions:
        vname = sess.get("video_name", "")
        group_id = derive_group_id(vname)
        task = sanitize_task_name(sess.get("taskType", ""))
        vmeta = sess.get("videoMetadata", {})
        duration = float(vmeta.get("duration", {}).get("seconds", 1e9))

        # Instructor utterances
        instructor_spans: List[Tuple[float, float, str]] = []
        for e in sess.get("events", []):
            if e.get("label") == "Conversation":
                attrs = e.get("attributes", {})
                cp = attrs.get("Conversation Purpose", "")
                if cp.startswith("instructor"):
                    u_s = float(e.get("start", 0.0))
                    u_e = float(e.get("end", 0.0))
                    u_text = attrs.get("Transcription", "")
                    instructor_spans.append((u_s, u_e, u_text))

        vpath = get_holoassist_video_path(vname, hd)
        has_video = vpath.exists()

        for idx, e in enumerate(sess.get("events", [])):
            if e.get("label") != "Fine grained action":
                continue
            actions_seen += 1
            attrs = e.get("attributes", {})
            ac = attrs.get("Action Correctness")
            if not ac:
                skip_counts["missing_correctness_attr"] += 1
                continue

            if ac == "Correct Action":
                label = 1
            elif ac.startswith("Wrong Action") or ac == "otherwise":
                label = 0
            else:
                skip_counts["missing_correctness_attr"] += 1
                continue

            a_s = float(e.get("start", 0.0))
            a_e = float(e.get("end", 0.0))
            if a_e <= a_s or a_s < 0:
                skip_counts["invalid_action_window"] += 1
                continue

            r_s = a_s
            r_e = min(duration, a_e + 5.0)
            if r_e <= r_s:
                skip_counts["invalid_reaction_window"] += 1
                continue

            if check_paths and not has_video:
                skip_counts["missing_video"] += 1
                continue

            # Reaction speech overlap & transcript
            overlapping_texts: List[str] = []
            spoke = 0
            for u_s, u_e, u_text in instructor_spans:
                if max(r_s, u_s) < min(r_e, u_e):
                    spoke = 1
                    if u_text and u_text.strip():
                        overlapping_texts.append(u_text.strip())

            transcript = " ".join(overlapping_texts) if overlapping_texts else ""

            verb = attrs.get("Verb", "").strip()
            noun = attrs.get("Noun", "").strip()
            context_text = f"Task: {task}. Step: {verb} {noun}."
            eid = e.get("id", idx)
            item_id = f"{vname}_{eid}_{idx}"

            item = Item(
                item_id=item_id,
                dataset="holoassist",
                group_id=group_id,
                video_path=str(vpath.resolve()),
                action_window_sec=[round(a_s, 3), round(a_e, 3)],
                reaction_window_sec=[round(r_s, 3), round(r_e, 3)],
                label=label,
                label_source="holoassist.Action Correctness",
                context_text=context_text,
                meta={
                    "spoke": spoke,
                    "transcript": transcript,
                    "session": vname,
                    "task": task,
                    "verb": verb,
                    "noun": noun,
                },
            )
            candidates.append(item)

    # 2. Partition groups 70/30
    unique_groups = sorted(list(set(c.group_id for c in candidates)))
    rng_group = np.random.default_rng(0)
    shuffled_groups = list(unique_groups)
    rng_group.shuffle(shuffled_groups)

    n_train_groups = int(round(0.7 * len(shuffled_groups)))
    train_groups = set(shuffled_groups[:n_train_groups])
    test_groups = set(shuffled_groups[n_train_groups:])

    train_cands = [c for c in candidates if c.group_id in train_groups]
    test_cands = [c for c in candidates if c.group_id in test_groups]

    # 3. Balanced sampling per split
    rng = np.random.default_rng(seed)

    # Train sampling
    tr_mistakes = [c for c in train_cands if c.label == 0]
    tr_correct = [c for c in train_cands if c.label == 1]
    n_tr_m = min(len(tr_mistakes), train_cap)
    n_tr_c = min(len(tr_correct), n_tr_m)

    if len(tr_mistakes) > n_tr_m:
        idx_m = rng.choice(len(tr_mistakes), size=n_tr_m, replace=False)
        idx_m.sort()
        sampled_tr_m = [tr_mistakes[i] for i in idx_m]
    else:
        sampled_tr_m = tr_mistakes

    if len(tr_correct) > n_tr_c:
        idx_c = rng.choice(len(tr_correct), size=n_tr_c, replace=False)
        idx_c.sort()
        sampled_tr_c = [tr_correct[i] for i in idx_c]
    else:
        sampled_tr_c = tr_correct[:n_tr_c]

    train_items = sampled_tr_m + sampled_tr_c

    # Test sampling
    te_mistakes = [c for c in test_cands if c.label == 0]
    te_correct = [c for c in test_cands if c.label == 1]
    n_te_m = min(len(te_mistakes), test_cap)
    n_te_c = min(len(te_correct), n_te_m)

    if len(te_mistakes) > n_te_m:
        idx_m = rng.choice(len(te_mistakes), size=n_te_m, replace=False)
        idx_m.sort()
        sampled_te_m = [te_mistakes[i] for i in idx_m]
    else:
        sampled_te_m = te_mistakes

    if len(te_correct) > n_te_c:
        idx_c = rng.choice(len(te_correct), size=n_te_c, replace=False)
        idx_c.sort()
        sampled_te_c = [te_correct[i] for i in idx_c]
    else:
        sampled_te_c = te_correct[:n_te_c]

    test_items = sampled_te_m + sampled_te_c

    all_items = train_items + test_items
    all_items.sort(key=lambda x: x.item_id)

    # Validate items
    errors = validate_items(all_items, check_paths=check_paths)
    if errors:
        raise ValueError(f"HoloAssist items validation failed with {len(errors)} errors: {errors[:5]}")

    stats = {
        "sessions_seen": len(sessions),
        "actions_seen": actions_seen,
        "actions_kept": len(all_items),
        "actions_skipped": skip_counts,
        "unique_groups": len(unique_groups),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "straddling_groups": 0,
        "train_items": len(train_items),
        "train_mistakes": len(sampled_tr_m),
        "train_correct": len(sampled_tr_c),
        "test_items": len(test_items),
        "test_mistakes": len(sampled_te_m),
        "test_correct": len(sampled_te_c),
    }

    split_map: Dict[str, str] = {
        it.item_id: ("train" if it.group_id in train_groups else "test")
        for it in all_items
    }

    return all_items, stats, split_map


def main() -> None:
    parser = argparse.ArgumentParser(description="HoloAssist source adapter and dataset builder")
    parser.add_argument("--stats", action="store_true", help="Print HoloAssist label statistics")
    parser.add_argument("--build", action="store_true", help="Build items, stats, and group split")
    parser.add_argument("--no-check-paths", action="store_true", help="Do not require video files to exist on disk")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    args = parser.parse_args()

    if args.stats:
        sessions = load_raw_annotations()
        splits = load_official_splits()
        stats = compute_stats(sessions, splits)
        print_stats(stats)
        return

    if args.build:
        print("Building HoloAssist items...")
        items, stats, split_map = build_holoassist_items(
            check_paths=not args.no_check_paths,
            seed=args.seed,
        )
        print(f"Built {len(items)} items ({stats['train_items']} train, {stats['test_items']} test).")

        out_dir = DATA_ROOT / "items" / "holoassist"
        out_dir.mkdir(parents=True, exist_ok=True)
        items_path = out_dir / "items.jsonl"
        stats_path = out_dir / "stats.json"

        write_items(items, items_path)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote {items_path} and {stats_path}")

        print("Generating group-disjoint split splits/holoassist.json...")
        split_data = make_group_split(
            items,
            "holoassist",
            official=split_map,
            source="grouped_70_30",
            seed=args.seed,
            force=True,
        )
        print(f"Split generated: {len(split_data['train'])} train, {len(split_data['test'])} test. sha256={split_data['sha256']}")


if __name__ == "__main__":
    main()
