# Layer 03d (Proxemic Kinematics) — 50-Clip Re-Run Smell Test

**Date**: June 13, 2026
**Goal**: Re-run the 50-clip smell test on current `main` — where all three June 9–11 issues are Resolved (window/cadence anchoring, SAM MPS float32, **span cap**) — to confirm the resolved state end-to-end and surface any *new* issue. Same 50-clip manifest as June 9/10 (climax metadata reused), so results are directly comparable.
**Host**: Mac Studio (M4 Max, 64 GB), MPS. Run under `tools/run_supervised.sh` (the silent-crash guard from the 03-arch supervised-runner work).

## Headline
| metric | June 10 (post-fix, pre-cap-live) | June 13 (current main) |
|---|---|---|
| clips scored | 48/50 | **35/50** |
| sentinels | 2 | **15** (13 span-capped + 2 single-detection) |
| person-task vectors | 112 | **79** (exactly the 112 − 33 long-span the cap predicted) |
| window source | 112/112 `bystander_anchored` | **79/79 `bystander_anchored`** |
| spans > 30 s (cap violations) | 33 (pre-cap) | **0** (max exactly 30.0 s) |
| SAM bbox-prompt failures | 0 | **0** |
| confidence-1.0 vectors | 10 (1 Approach, 1 Avoidance, 8 sub-thr Neutral) | **10 (identical set)** |
| chaos-rejected (noise > 15) | 93/112 | 60/79 (76 %) |
| runtime | ~24 min (w/ crash+resume) | **793 s, single clean attempt** |

The run is a **faithful reproduction of the June 10 signal** (same 10 confident vectors, byte-identical exemplar magnitudes) **plus end-to-end confirmation of the span cap**: precisely the 33 long-span vectors are gone, 0 windows exceed 30 s, the max sits exactly on the 30.0 s boundary, and the 10 confident vectors (all on ≤ 12 s spans) are retained untouched. SAM ran on real masks with 0 failures. No crash this time; the supervisor exited clean on attempt 1.

## Why scored dropped 48 → 35 (this is the cap working, not a regression)
The span cap (Resolved #3) became live between the runs. It returns `None` for any bystander whose anchored window exceeds 30 s, and when *all* of a clip's bystanders are skipped the clip becomes a sentinel. Replaying the actual `_bystander_measurement_window` against the manifest confirms all 15 sentinels are legitimate: **13 span-capped** (anchored windows 33–198 s, every one genuinely > 30 s) + **2 single-detection** (`51cb7800`, `11fc65a9` — no delta measurable, same as June 10). The 33 removed vectors were all chaos-rejected zeros on June 10, so **no real signal was lost.**

## Spot-check (visual) — the two top signals split
- **Approach `343f4d2d`**, **v = +0.41**, conf 1.0 (bbox +28.8 %, depth −0.153, win [39,45]): boxed pedestrian in a dark shirt is clearly larger/closer at t=45 s than t=39 s as the wearer closes distance on a night street. Box tracks the same person; window start→end IoU 0.52. ✅ **Genuine.** (`frames/approach_*_343f4d2d.jpg`)
- **Avoidance `599f2f09`**, **v = −0.48**, conf 1.0 (bbox −80.8 %, depth +0.063, win [9,21]): **tracking ID-switch, not a recoil.** `person_id=2` is the man holding a large banner at t=9 s (area 79,588 px) but the same id is on a **different, distant child** at t=21 s (area 15,246 px), while the banner-holder walked off frame-right. The −80.8 % "shrink" is the box jumping between two bodies. Window start→end IoU **0.00**. ❌ **Confident false positive.** (`frames/avoidance_*_599f2f09.jpg`, `frames/599f2f09_track/`)
- **Failed-to-score** sentinels verified legitimate (above).

The June 10 report misread this exemplar as "festival-goer walking away" — the re-spot-check caught the ID-switch. The numbers are identical; only the interpretation changed, which is exactly the value of re-checking.

## ID-switch prevalence (quantified)
Across the 15 clean non-zero vectors, 4 carry the discontinuity signature (start→end box IoU ≈ 0.00 + centroid jump > 0.5× diag); genuine vectors all sit at **IoU ≥ 0.31** (clean separation — Approach exemplar is 0.52). Cross-referencing box area ratio:
| clip · pid | vec | conf | IoU | areaRatio | verdict |
|---|---|---|---|---|---|
| `599f2f09` · 2 | −0.48 | 1.0 | 0.00 | 0.19 | **harmful switch — UNCAUGHT** |
| `18323b66` · 0 | +0.30 | 0.0 | 0.00 | 1.88 | harmful switch, **caught** by confidence gate (depth disagreed) |
| `599f2f09` · 0 | +0.28 | 1.0 | 0.00 | 1.00 | benign — same-area position jump |
| `10167fcf` · 0 | −0.09 | 1.0 | 0.00 | 0.96 | benign — camera pan, same person |

So **exactly one** confident, uncaught, harmful artifact — and it is the run's headline Avoidance. Neither gate catches it: the confidence gate only checks bbox/depth sign agreement (a near→far switch passes), and the chaos gate passed it (noise 14.19 < 15).

## New issues filed (docs/03d Unresolved, for selection)
1. **Issue 1 — `person_id` discontinuity → confident false vectors** (the −0.48 above). Recommended: an **identity-continuity guard** (reject when consecutive in-window box IoU ≈ 0 *and* large area-ratio change), which the clean IoU separation makes surgical (catches the switch, spares benign pans + all genuine vectors).
2. **Issue 2 — climax-gap anchoring → 30 % sentinel rate**. The ±1-detection anchor lands a > 30 s window whenever a bystander has a detection gap at the climax (10/17 capped bystanders are densely tracked elsewhere). Honest "no data," not a false positive, but it changed the headline yield. Recommended: an explicit `no_detection_near_climax` sentinel reason for legibility.

## Verdict
- **The Resolved fixes hold end-to-end**: span cap confirmed live (0 violations, 33 long-span vectors removed exactly as projected), SAM masks on MPS with 0 failures, supervisor clean single-pass, signal reproduced.
- **But the smell test caught a real correctness defect**: 03d emits a confident false Avoidance from an upstream ID-switch its gates can't see. Of the two above-threshold actions, Approach is genuine and Avoidance is an artifact. Filed as docs/03d Unresolved Issue 1.
- The 30 % sentinel rate is the honest cost of the span cap on sparse-at-climax tracks (Issue 2), not a regression.

## Artifacts
`manifest_03d_50.json`, `03d_result_50.json`, `run_03d_50.log`, `supervise.log`, `analyze.py`, `run.py`, `frames/approach_*`, `frames/avoidance_*`, `frames/599f2f09_track/` (the ID-switch), `frames/idswitch_check/` (10167fcf benign pan).
