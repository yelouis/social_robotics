"""SRB v0 — deterministic item templating from golden labels (docs/07 §5).

F1 (valence 4-way) and F2 (credit assignment 3-way), Track A; F3 via
pair_items.py; Track B via render_track_b.py. No LLM writes items (v0/v1).

Eligibility + quotas (§5.1/5.3), dedup (§5.4.2). Items carry gold in a
separate sealed file (answers never ship with public questions).

Usage: python bench/build_items.py [--seed 5]
Outputs: bench_v0/items/items_f{1,2}_track_a.jsonl (+ sealed gold_f{1,2}.jsonl)
"""
import argparse
import random
from collections import Counter, defaultdict

from srb_common import BENCH_DATA, PILOT_WAVE, read_jsonl, write_jsonl

F1_OPTIONS = ["Approving / positive", "Disapproving / negative",
              "Neutral acknowledgment — noticed, no evaluative response",
              "No reaction directed at the wearer"]
F2_OPTIONS = ["Yes — it responds to the wearer's action",
              "No — it responds to something or someone else",
              "Not enough evidence to tell"]
F1_GOLD = {"approving": 0, "disapproving": 1, "neutral": 2, "no_reaction": 3}
F2_GOLD = {"wearer": 0, "something_else": 1, "cant_tell": 2}


def gerundize(caption: str) -> str:
    """'points at the signboard' -> 'is pointing at the signboard' (best-effort);
    sentinels handled by callers."""
    words = caption.strip().rstrip(".").split()
    if not words:
        return caption
    v = words[0].lower()
    if v.endswith("ing"):
        ger = v
    elif v.endswith("s") and not v.endswith("ss"):
        stem = v[:-1]
        ger = (stem[:-1] + "ing") if stem.endswith("e") and not stem.endswith("ee") else stem + "ing"
    else:
        ger = v + "ing"
    return " ".join(["is", ger] + words[1:])


def f1_gold_class(g):
    """§5.1: gold from B5; 'no reaction' from B2=yes ∧ B3=no, or B4=something_else."""
    if g["reaction"] == "no" and g["audience"] == "yes":
        return "no_reaction"
    if g["directedness"] == "something_else":
        return "no_reaction"
    if g["directedness"] == "wearer" and g["valence"] in ("approving", "disapproving", "neutral"):
        return g["valence"]
    return None          # mixed/ambiguous or unsure chains — excluded from F1


def build(golden, moments, seed):
    rng = random.Random(seed)
    by_moment = {m["moment_id"]: m for m in moments}
    used_pairs = defaultdict(set)     # family -> {(clip, frozenset(bystanders))} dedup §5.4.2

    def bystanders(mid):
        pf = by_moment.get(mid, {}).get("engine_prefill") or {}
        return frozenset(p.get("person_id") for p in pf.get("per_person", [])) or frozenset([None])

    def dedup_ok(family, mid):
        key = (by_moment.get(mid, {}).get("clip_id"), bystanders(mid))
        if key in used_pairs[family]:
            return False
        used_pairs[family].add(key)
        return True

    f1_pool = []
    for g in golden:
        cap = g["wearer_action"]
        if cap.lower() == "unclear":
            continue                                  # excluded from all action-conditioned families
        cls = f1_gold_class(g)
        if cls is None:
            continue
        f1_pool.append((g, cls))

    # §5.3 quotas: each class >=15%; no_reaction 20-30%. Enforce by downsampling
    # the overrepresented classes toward feasibility (never fabricates items).
    by_cls = defaultdict(list)
    for g, cls in f1_pool:
        by_cls[cls].append(g)
    for v in by_cls.values():
        rng.shuffle(v)
    counts = {c: len(v) for c, v in by_cls.items()}
    n_total = sum(counts.values())
    quota_note = None
    if counts and n_total:
        # cap no_reaction at 30% of final set; require every present class >=15%
        min_cls = min(counts.values())
        max_total_by_min = int(min_cls / 0.15) if min_cls else n_total
        nr = counts.get("no_reaction", 0)
        max_total_by_nr = int(nr / 0.20) if nr else max_total_by_min
        target = min(n_total, max_total_by_min, max_total_by_nr)
        if target < n_total:
            quota_note = f"downsampled {n_total}->{target} for class quotas"
        keep = []
        for c, v in by_cls.items():
            cap_c = int(target * 0.30) if c == "no_reaction" else target
            keep.extend((g, c) for g in v[:max(1, cap_c)])
        f1_pool = keep[:target] if target else keep

    items_f1, gold_f1 = [], []
    for g, cls in f1_pool:
        mid = g["moment_id"]
        if not dedup_ok("f1", mid):
            continue
        cap = g["wearer_action"]
        ctx = ("The camera wearer is in a conversation" if cap.lower() == "conversation"
               else f"The camera wearer {gerundize(cap)}")
        iid = f"{PILOT_WAVE}-f1-{len(items_f1):04d}"
        items_f1.append({
            "item_id": iid, "family": "F1", "track": "A", "wave": PILOT_WAVE,
            "moment_id": mid, "clip": f"kits/{mid}/clip_s.mp4",
            "question": (f"{ctx}. Based on the visible people's response in the clip, "
                         "what social feedback did the wearer's action receive?"),
            "options": F1_OPTIONS,
        })
        gold_f1.append({"item_id": iid, "gold_index": F1_GOLD[cls], "gold_class": cls,
                        "moment_id": mid, "channels": g["channels"]})

    items_f2, gold_f2 = [], []
    for g in golden:
        if g["reaction"] != "yes" or not g["directedness"]:
            continue
        mid = g["moment_id"]
        if not dedup_ok("f2", mid):
            continue
        iid = f"{PILOT_WAVE}-f2-{len(items_f2):04d}"
        items_f2.append({
            "item_id": iid, "family": "F2", "track": "A", "wave": PILOT_WAVE,
            "moment_id": mid, "clip": f"kits/{mid}/clip_s.mp4",
            "question": ("A person in the clip reacts during this moment. "
                         "Is their reaction a response to the camera wearer?"),
            "options": F2_OPTIONS,
        })
        gold_f2.append({"item_id": iid, "gold_index": F2_GOLD[g["directedness"]],
                        "gold_class": g["directedness"], "moment_id": mid,
                        "channels": g["channels"]})

    # §5.3 F2 quota check (reported, not enforced by fabrication)
    f2_dist = Counter(g["gold_class"] for g in gold_f2)
    n2 = max(1, len(gold_f2))
    f2_minority = (f2_dist.get("something_else", 0) + f2_dist.get("cant_tell", 0)) / n2

    report = {
        "f1_items": len(items_f1), "f1_classes": dict(Counter(g["gold_class"] for g in gold_f1)),
        "f1_quota_note": quota_note,
        "f2_items": len(items_f2), "f2_classes": dict(f2_dist),
        "f2_minority_share": round(f2_minority, 3),
        "f2_quota_met_(>=0.35)": f2_minority >= 0.35,
    }
    return items_f1, gold_f1, items_f2, gold_f2, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    golden = read_jsonl(BENCH_DATA / "golden_labels.jsonl")
    moments = read_jsonl(BENCH_DATA / "candidate_moments.jsonl")
    i1, g1, i2, g2, report = build(golden, moments, a.seed)
    out = BENCH_DATA / "items"
    write_jsonl(out / "items_f1_track_a.jsonl", i1)
    write_jsonl(out / "items_f2_track_a.jsonl", i2)
    write_jsonl(out / "gold_f1.jsonl", g1)          # SEALED — never publish
    write_jsonl(out / "gold_f2.jsonl", g2)
    import json
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
