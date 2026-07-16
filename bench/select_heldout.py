"""SRB v0 — Ring-1 held-out selection + wearer-grouped split stamping.

Implements the docs/06 Issue-6 blocking prerequisite in its minimal
benchmark-owned form (docs/07 §7 v0 step 1): a SPLITS FILE, not a manifest
mutation. Every Ego4D clip is assigned to exactly one of:

  published_train   — in the published 991 (or shares a WEARER with it):
                      may never source a test item (docs/07 §2 label leakage)
  test_heldout      — wearer-disjoint from every published clip; the v0 pool

Selection of NEW download candidates (not yet local): wearer-disjoint,
scenario-weighted toward scenarios that historically yielded social segments
(frequency among the published 991 — those survived the social-presence
filter), duration-bounded, capped per wearer for diversity.

Usage:
    python bench/select_heldout.py --target-clips 40 [--min-sec 300] [--max-sec 2400]
Outputs:
    bench/splits_ego4d.json      (tracked)  — uid -> split for every known uid
    bench_v0/heldout_download_uids.json     — the download list for harvest_heldout.py
"""
import argparse
import json
import random
from collections import Counter, defaultdict

from srb_common import EGO4D_META, PUBLISHED_MANIFEST, SPLITS_FILE, BENCH_DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-clips", type=int, default=40)
    ap.add_argument("--min-sec", type=float, default=300)
    ap.add_argument("--max-sec", type=float, default=2400)
    ap.add_argument("--max-per-wearer", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    with open(EGO4D_META) as f:
        videos = json.load(f)["videos"]
    by_uid = {v["video_uid"]: v for v in videos}
    with open(PUBLISHED_MANIFEST) as f:
        pub_ids = {e.get("id", e.get("video_id")) for e in json.load(f)}

    pub_wearers = {by_uid[u]["fb_participant_id"] for u in pub_ids if u in by_uid}
    scen_weight = Counter(s for u in pub_ids if u in by_uid
                          for s in (by_uid[u].get("scenarios") or []))

    splits = {}
    for u in pub_ids:
        splits[u] = "published_train"

    # Every non-published clip of a published wearer is train-only by grouping.
    candidates = []
    per_wearer = defaultdict(int)
    for v in videos:
        u = v["video_uid"]
        if u in splits:
            continue
        w = v.get("fb_participant_id")
        if w in pub_wearers:
            splits[u] = "published_train"     # wearer-grouped exclusion
            continue
        splits[u] = "test_heldout"
        dur = v.get("duration_sec") or 0
        if not (a.min_sec <= dur <= a.max_sec):
            continue
        weight = max((scen_weight.get(s, 0) for s in (v.get("scenarios") or [])), default=0)
        if weight == 0:
            continue                          # scenario never yielded social clips
        candidates.append((weight, rng.random(), u, w, dur))

    # Rank: social-yield weight desc, random tiebreak; cap per wearer.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    picked = []
    for weight, _, u, w, dur in candidates:
        if per_wearer[w] >= a.max_per_wearer:
            continue
        per_wearer[w] += 1
        picked.append({"video_uid": u, "wearer": w, "duration_sec": dur,
                       "scenario_weight": weight,
                       "scenarios": by_uid[u].get("scenarios")})
        if len(picked) >= a.target_clips:
            break

    SPLITS_FILE.write_text(json.dumps(
        {"generated_from": "select_heldout.py",
         "rule": "wearer-grouped: any clip sharing fb_participant_id with the published 991 is published_train",
         "n_published_train": sum(1 for s in splits.values() if s == "published_train"),
         "n_test_heldout": sum(1 for s in splits.values() if s == "test_heldout"),
         "splits": splits}, indent=1))

    BENCH_DATA.mkdir(parents=True, exist_ok=True)
    out = BENCH_DATA / "heldout_download_uids.json"
    out.write_text(json.dumps(picked, indent=2))
    print(f"[select_heldout] splits: {sum(1 for s in splits.values() if s=='published_train')} published_train "
          f"/ {sum(1 for s in splits.values() if s=='test_heldout')} test_heldout -> {SPLITS_FILE}")
    print(f"[select_heldout] download list: {len(picked)} clips, "
          f"{len({p['wearer'] for p in picked})} wearers, "
          f"{sum(p['duration_sec'] for p in picked)/3600:.1f} h -> {out}")


if __name__ == "__main__":
    main()
