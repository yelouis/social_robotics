# SocialRobotics-Bench — v0 pilot toolchain

Implements docs/07 §7 v0 ("contact-sheet benchmark"). Everything below runs
from the repo root with the project venv. All working data lives on the SSD
at `bench_v0/` (Ego4D pixels → **internal-only**); the only tracked outputs
are `bench/splits_ego4d.json` and, eventually, the public Track-B wave.

## Pipeline at a glance

```
select_heldout ─▶ harvest (download+Node-02) ─▶ 02b/03x pre-pass ─▶ export_candidates
      │                                                                  │
splits_ego4d.json                                              candidate_moments.jsonl
                                                                         │
                 make_rating_kit ─▶ YOU RATE (CSV) ─▶ adjudicate ─▶ golden_labels.jsonl
                                                                         │
                     build_items + pair_items + render_track_b + blind_gate ─▶ items/
                                                                         │
                                                              score.py (per model)
```

## Step 0 — automated pre-pass (launched as a daemon; hours)

```zsh
venv/bin/python bench/select_heldout.py --target-clips 40   # splits + download list
zsh bench/run_engine_prepass.sh                              # download → Node-02 → 02b → 03x → kits
```

When it finishes you have `bench_v0/kits/…` (one folder per moment:
`clip_s.mp4`, `clip_splus.mp4`, `strip.jpg`), `kit_manifest.csv`, and an empty
`ratings_maintainer.csv`.

## Step 1 — YOUR PART: rate (≈4–6 h total, splittable)

1. Read `bench/LABELING_GUIDE.md` once (10 min).
2. Open `bench_v0/ratings_maintainer.csv` in a spreadsheet next to the kits
   folder. For each row: watch `clip_s.mp4` (+ `clip_splus.mp4` for B7),
   fill A1–B9 with the short codes. Rough `seconds_spent` per item is gold
   for the v1 tooling decision — please fill it when you remember.
3. Validate in batches (catches typos early):
   `venv/bin/python bench/adjudicate.py --validate "/Volumes/Extreme SSD/social_robotics/bench_v0/ratings_maintainer.csv"`

## Step 2 — retest sample (same day it finishes)

```zsh
venv/bin/python bench/adjudicate.py --make-retest ".../ratings_maintainer.csv"
```

**Wait ≥3 days (washout).** Then:

```zsh
venv/bin/python bench/make_rating_kit.py --only-retest ".../bench_v0/retest_ids.json"
```

and blind re-rate `ratings_maintainer_retest.csv` (fresh order; don't look at
round one).

## Step 3 — finalize golden labels + the v0 gate

```zsh
venv/bin/python bench/adjudicate.py --finalize ".../ratings_maintainer.csv" \
    --retest ".../ratings_maintainer_retest.csv"
```

Emits `golden_labels.jsonl` + `self_consistency_report.json`. **Go/no-go**:
≥80 % self-consistency on B3/B4/B5 — below it, we revise the instrument and
re-rate, per spec (never the gate).

## Step 4 — items, gate, anchors (automated)

```zsh
venv/bin/python bench/build_items.py          # F1/F2 + quotas + dedup
venv/bin/python bench/pair_items.py           # F3 pairs (if valence spread allows)
venv/bin/python bench/render_track_b.py       # Track-B (publishable) variants
venv/bin/python bench/blind_gate.py           # kills text-prior-answerable items
```

Second gate: blind-survival ≥50 % (`blind_gate_report.json`).

Score any model: produce `{"item_id","answer"}` JSONL, then
`venv/bin/python bench/score.py preds.jsonl --label <model>` — reports
per-family micro/macro accuracy, **SHR** (social-hallucination rate), and the
Track A/B delta.

## Rules that bind everything (docs/07)

- Kits and Track-A clips are Ego4D pixels: **never publish, never upload**.
  Track-B items + golden labels (sans clip pixels) are the publishable pilot.
- Engine outputs are never gold; raters never see them (kits omit
  `engine_prefill` by construction).
- Pilot artifacts must say: *single-annotator golden labels (project author)*
  — dev-grade, not citable gold (that starts at v1's 2-rater regime).
