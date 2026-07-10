"""SRB v0 — validate rating CSVs and emit golden labels (docs/07 §4.5,
single-rater regime).

Modes:
  --validate ratings.csv
      Enum + skip-logic check only (run early and often while rating).
  --make-retest ratings.csv
      Sample the random 20 % of ACCEPTED moments for the ≥3-day-washout blind
      re-rate; writes retest_ids.json (feed to make_rating_kit --only-retest).
  --finalize ratings.csv [--retest ratings_retest.csv]
      Apply the v0 rules and emit golden_labels.jsonl + self-consistency report:
        * triage: A1=yes ∧ A2=yes; any A3 flag -> quarantined (listed, excluded)
        * B8=guessing -> dropped from test candidacy
        * retest disagreement on B3/B4/B5 -> dropped from test candidacy
        * gate: ≥80 % exact match on B3/B4/B5 across the retest sample

The multi-rater merge/κ path is v1 (docs/07 §7).
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter

from srb_common import (BENCH_DATA, ENUMS, B6_CHANNELS, A3_FLAGS,
                        RATING_COLUMNS, write_jsonl)

GATE_FIELDS = ("B3", "B4", "B5")
RETEST_FRACTION = 0.20
GATE_THRESHOLD = 0.80


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def norm(v):
    return (v or "").strip().lower().replace(" ", "_")


def validate(rows):
    """Return (clean_rows, errors). Skip-logic (§4.4): B3 in (no, unsure) ->
    B4-B6 must be blank; B4 != wearer -> B5 blank; B7 only when B3=yes."""
    errors, clean = [], []
    seen = set()
    for i, r in enumerate(rows, 2):          # header = line 1
        mid = r.get("moment_id", "").strip()
        e = []
        if not mid:
            continue                          # unrated template row
        if mid in seen:
            e.append("duplicate moment_id")
        seen.add(mid)
        row = {k: norm(r.get(k)) for k in RATING_COLUMNS}
        row["moment_id"] = mid
        row["B1"] = (r.get("B1") or "").strip()          # free text, keep case
        row["B9"], row["C1"] = (r.get("B9") or "").strip(), (r.get("C1") or "").strip()
        # Multi-value fields: split on ';' FIRST, then normalize each token
        # (wholesale norm() would fuse 'face; head_gesture' -> '_head_gesture').
        for multi in ("A3", "B6"):
            toks = [norm(t) for t in (r.get(multi) or "").split(";")]
            row[multi] = ";".join(t for t in toks if t)
        if not row["A1"]:
            continue                          # untouched row — not yet rated
        for f_ in ("A1", "A2"):
            if row[f_] not in ENUMS[f_]:
                e.append(f"{f_}={row[f_]!r} not in {ENUMS[f_]}")
        for flag in filter(None, (x.strip() for x in row["A3"].split(";"))):
            if flag not in A3_FLAGS:
                e.append(f"A3 flag {flag!r} not in {A3_FLAGS}")
        triaged_out = row["A1"] != "yes" or row["A2"] != "yes"
        if not triaged_out:
            if not row["B1"]:
                e.append("B1 required (use 'conversation'/'unclear' sentinels)")
            for f_ in ("B2", "B3", "B8"):
                if row[f_] not in ENUMS[f_]:
                    e.append(f"{f_}={row[f_]!r} not in {ENUMS[f_]}")
            if row["B3"] == "yes":
                if row["B4"] not in ENUMS["B4"]:
                    e.append(f"B4={row['B4']!r} required when B3=yes")
                if row["B4"] == "wearer" and row["B5"] not in ENUMS["B5"]:
                    e.append(f"B5={row['B5']!r} required when B4=wearer")
                if row["B4"] != "wearer" and row["B5"]:
                    e.append("B5 must be blank when B4!=wearer (skip logic)")
                for ch in filter(None, (x.strip() for x in row["B6"].split(";"))):
                    if ch not in B6_CHANNELS:
                        e.append(f"B6 channel {ch!r} not in {B6_CHANNELS}")
                if row["B7"] and row["B7"] not in ENUMS["B7"]:
                    e.append(f"B7={row['B7']!r} not in {ENUMS['B7']}")
            else:
                for f_ in ("B4", "B5", "B7"):
                    if row[f_]:
                        e.append(f"{f_} must be blank when B3={row['B3']} (skip logic)")
        if e:
            errors.append((i, mid, e))
        else:
            clean.append(row)
    return clean, errors


def accepted(rows):
    return [r for r in rows if r["A1"] == "yes" and r["A2"] == "yes" and not r["A3"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate")
    ap.add_argument("--make-retest")
    ap.add_argument("--finalize")
    ap.add_argument("--retest")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    if a.validate:
        rows, errors = validate(load(a.validate))
        for line, mid, errs in errors:
            print(f"  line {line} ({mid}): {'; '.join(errs)}")
        print(f"[validate] {len(rows)} valid rated rows, {len(errors)} rows with errors")
        sys.exit(1 if errors else 0)

    if a.make_retest:
        rows, errors = validate(load(a.make_retest))
        if errors:
            sys.exit(f"[retest] fix {len(errors)} validation errors first (--validate)")
        acc = accepted(rows)
        n = max(1, round(len(acc) * RETEST_FRACTION))
        ids = [r["moment_id"] for r in random.Random(a.seed).sample(acc, n)]
        out = BENCH_DATA / "retest_ids.json"
        out.write_text(json.dumps(ids, indent=1))
        print(f"[retest] {n}/{len(acc)} accepted moments sampled -> {out}")
        print("[retest] wait >= 3 days, then: python bench/make_rating_kit.py "
              f"--only-retest {out}")
        return

    if a.finalize:
        rows, errors = validate(load(a.finalize))
        if errors:
            sys.exit(f"[finalize] fix {len(errors)} validation errors first (--validate)")
        by_id = {r["moment_id"]: r for r in rows}
        acc = accepted(rows)
        quarantined = [r["moment_id"] for r in rows if r["A3"]]

        consistency = None
        disagree = set()
        if a.retest:
            re_rows, re_err = validate(load(a.retest))
            if re_err:
                sys.exit(f"[finalize] retest CSV has {len(re_err)} validation errors")
            matches = Counter()
            total = 0
            for rr in re_rows:
                orig = by_id.get(rr["moment_id"])
                if not orig:
                    continue
                total += 1
                row_match = all(orig[f_] == rr[f_] for f_ in GATE_FIELDS)
                for f_ in GATE_FIELDS:
                    matches[f_] += orig[f_] == rr[f_]
                if not row_match:
                    disagree.add(rr["moment_id"])
            consistency = {f_: round(matches[f_] / total, 3) for f_ in GATE_FIELDS} if total else {}
            consistency["all_fields_row_level"] = round((total - len(disagree)) / total, 3) if total else None
            consistency["n_retested"] = total

        golden, dropped = [], Counter()
        for r in acc:
            if r["B8"] == "guessing":
                dropped["guessing"] += 1
                continue
            if r["moment_id"] in disagree:
                dropped["retest_disagreement"] += 1
                continue
            golden.append({
                "moment_id": r["moment_id"],
                "wearer_action": r["B1"],
                "audience": r["B2"],
                "reaction": r["B3"],
                "directedness": r["B4"] or None,
                "valence": r["B5"] or None,
                "channels": [c for c in r["B6"].split(";") if c],
                "next_action": r["B7"] or None,
                "confidence": r["B8"],
                "notes": r["B9"] or None,
                "rater": "maintainer",
                "regime": "v0_single_rater",
            })
        write_jsonl(BENCH_DATA / "golden_labels.jsonl", golden)

        report = {
            "n_rated": len(rows), "n_accepted_triage": len(acc),
            "n_quarantined_a3": len(quarantined), "quarantined": quarantined,
            "n_golden": len(golden), "dropped": dict(dropped),
            "self_consistency": consistency,
            "gate_pass": (None if consistency is None else
                          all(consistency.get(f_, 0) >= GATE_THRESHOLD for f_ in GATE_FIELDS)),
            "label_distribution": {
                "B3": dict(Counter(r["reaction"] for r in golden)),
                "B4": dict(Counter(r["directedness"] for r in golden if r["directedness"])),
                "B5": dict(Counter(r["valence"] for r in golden if r["valence"])),
            },
        }
        (BENCH_DATA / "self_consistency_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if consistency is None:
            print("[finalize] NOTE: no retest CSV supplied — golden labels emitted "
                  "but the v0 gate (>=80% on B3/B4/B5) is UNMEASURED.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
