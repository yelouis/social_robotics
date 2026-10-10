from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DATA_ROOT


def get_labels_dir() -> Path:
    return DATA_ROOT / "raw" / "holoassist" / "labels"


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
            prefix = s_id.split("-")[0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="HoloAssist source adapter and statistics")
    parser.add_argument("--stats", action="store_true", help="Print HoloAssist label statistics")
    args = parser.parse_args()

    if args.stats:
        sessions = load_raw_annotations()
        splits = load_official_splits()
        stats = compute_stats(sessions, splits)
        print_stats(stats)


if __name__ == "__main__":
    main()
