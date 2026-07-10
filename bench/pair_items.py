"""SRB v0 — F3 pairwise-preference items (docs/07 §5.1 pairing rules).

Rules: same action category (v0 verb-cluster = first verb token of B1),
different clips, valences strictly ordered and NON-ADJACENT in rank
(approving > neutral > no_reaction > disapproving), presentation order
randomized, each moment in <= 2 pairs.

Usage: python bench/pair_items.py [--seed 5]
"""
import argparse
import random
from collections import defaultdict

from srb_common import BENCH_DATA, PILOT_WAVE, read_jsonl, write_jsonl
from build_items import f1_gold_class, gerundize

RANK = {"approving": 0, "neutral": 1, "no_reaction": 2, "disapproving": 3}


def verb_cluster(caption: str):
    w = caption.strip().rstrip(".").split()
    if not w:
        return None
    v = w[0].lower()
    return v[:-1] if (v.endswith("s") and not v.endswith("ss")) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    golden = read_jsonl(BENCH_DATA / "golden_labels.jsonl")
    moments = {m["moment_id"]: m for m in read_jsonl(BENCH_DATA / "candidate_moments.jsonl")}

    pool = []
    for g in golden:
        cap = g["wearer_action"]
        if cap.lower() in ("unclear", "conversation"):
            continue                                  # F3 excludes both sentinels (docs/06 case 1)
        cls = f1_gold_class(g)
        if cls is None:
            continue
        pool.append((verb_cluster(cap), cls, g))

    by_verb = defaultdict(list)
    for verb, cls, g in pool:
        if verb:
            by_verb[verb].append((cls, g))

    items, gold, uses = [], [], defaultdict(int)
    for verb, entries in sorted(by_verb.items()):
        rng.shuffle(entries)
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                (c1, g1), (c2, g2) = entries[i], entries[j]
                if abs(RANK[c1] - RANK[c2]) < 2:
                    continue                          # non-adjacent ranks only (v0)
                m1, m2 = g1["moment_id"], g2["moment_id"]
                if moments.get(m1, {}).get("clip_id") == moments.get(m2, {}).get("clip_id"):
                    continue                          # never two moments of one clip
                if uses[m1] >= 2 or uses[m2] >= 2:
                    continue
                better = g1 if RANK[c1] < RANK[c2] else g2
                other = g2 if better is g1 else g1
                first, second = (better, other) if rng.random() < 0.5 else (other, better)
                iid = f"{PILOT_WAVE}-f3-{len(items):04d}"
                items.append({
                    "item_id": iid, "family": "F3", "track": "A", "wave": PILOT_WAVE,
                    "moment_ids": [first["moment_id"], second["moment_id"]],
                    "clips": [f"kits/{first['moment_id']}/clip_s.mp4",
                              f"kits/{second['moment_id']}/clip_s.mp4"],
                    "action_category": verb,
                    "question": ("Clips 1 and 2 each show the camera wearer "
                                 f"{gerundize(better['wearer_action'])} (or a similar action). "
                                 "In which clip did the wearer's action receive the more "
                                 "positive social response?"),
                    "options": ["Clip 1", "Clip 2"],
                })
                gold.append({"item_id": iid,
                             "gold_index": 0 if first is better else 1,
                             "moment_ids": [first["moment_id"], second["moment_id"]]})
                uses[m1] += 1
                uses[m2] += 1

    write_jsonl(BENCH_DATA / "items" / "items_f3_track_a.jsonl", items)
    write_jsonl(BENCH_DATA / "items" / "gold_f3.jsonl", gold)
    print(f"[pair] {len(items)} F3 pairs from {len(pool)} eligible moments "
          f"across {len(by_verb)} verb clusters")


if __name__ == "__main__":
    main()
