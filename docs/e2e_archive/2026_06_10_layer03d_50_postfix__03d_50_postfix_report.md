# Layer 03d (Proxemic Kinematics) — 50-Clip Post-Fix Smell Test

**Date**: June 10, 2026
**Goal**: Re-run the 50-clip smell test after resolving the two June 9 issues (per the user's selections): **Issue 1 Option A** (per-bystander window anchoring + detection-span widening) and **Issue 2** (SAM MPS float64 → float32). Verify the 0/50 dead-yield is fixed and that depth runs on real SAM masks.
**Host**: Mac Studio (M4 Max, 64 GB), MPS. Same 50-clip manifest as June 9 (climax metadata reused).

## Headline: 0/50 → 48/50 scored
| metric | June 9 (pre-fix) | June 10 (post-fix) |
|---|---|---|
| clips scored | **0/50** (all sentinels) | **48/50** (2 legitimate sentinels) |
| person-task vectors | 0 | **112** |
| window source | — | **112/112 `bystander_anchored`** |
| SAM bbox-prompt failures | every frame (silent bbox fallback) | **0** |
| non-zero-confidence vectors | 0 | 10 (1 Approach, 1 Avoidance, 8 sub-threshold Neutral) |
| chaos-rejected (noise > 15) | — | 93/112 (honest egocentric-shake rejection) |
| runtime | 181 s (nothing ran) | ~24 min of compute for the full 50 |

**Every one of the 112 vectors required the anchored window** — not a single bystander had ≥ 2 detections inside a strict 2 s reaction window, which retro-confirms the June 9 geometry diagnosis exactly (6 s detection cadence vs 2 s windows).

## Fix 1 — per-bystander anchoring + span (Issue 1, Option A)
`_bystander_measurement_window()` keeps the strict `task_reaction_window_sec` when it already holds ≥ 2 detections (dense-detection behavior unchanged, e.g. synthetic tests), otherwise anchors to the climax-nearest detection ± `ANCHOR_SPAN_DETECTIONS` (=1) — a 2–3-detection span, median **15 s** on this corpus. Ego-motion noise is now measured over each bystander's actual measurement window (cached per window), and every record carries `measurement_window_sec` + `window_source` provenance.

## Fix 2 — SAM float64 → float32 (Issue 2)
`SamProcessor` float64 tensors are cast to float32 before `.to('mps')`; integer size tensors untouched; a one-time **loud warning** fires if SAM ever falls back. Validated single-frame before the run: mask = 146 k px silhouette vs the 569 k px raw bbox, masked depth median **193 vs 72** — i.e. the bbox fallback had been diluting the person's depth with dark background, and the run log shows **0 SAM failures across all 50 clips**.

## Spot-check (visual) — both exemplars make sense
- **Approach** `343f4d2d` "Walking on street", **v = +0.41** (bbox +28.8 %, depth −0.153, win [39, 45]): green-boxed pedestrian directly ahead of the wearer grows 7.8 % → 10.1 % of frame as the wearer closes distance at night. ✅ (`frames/approach_*_343f4d2d.jpg`)
- **Avoidance** `599f2f09` "Attending a festival or fair", **v = −0.48** (bbox −80.8 %, depth +0.063, win [9, 21]): boxed festival-goer shrinks 5.1 % → 1.0 % of frame, walking away past the tent. ✅ (`frames/avoidance_*_599f2f09.jpg`)
- **Failed-to-score** `51cb7800` / `11fc65a9`: legitimate sentinels — every bystander in both clips has only a **single** detection, so no delta is measurable from the manifest at all.
- Sign convention holds across all 10 non-zero vectors: positive pairs with (bbox growth, depth decrease), negative with (bbox shrink, depth increase).

## Honest-yield profile
93/112 vectors are zeroed by the ego-motion chaos gate (`optical_flow_noise > 15`) — expected on egocentric footage where the camera itself shakes; the gate is doing its documented job, mirroring the 03b/03c honesty pattern. 10 vectors survive with confidence 1.0 and physically consistent magnitudes.

## Residual finding (filed as new docs/03d Unresolved Issue 1)
The anchored span is **unbounded**: with sparse tracks the ±1-detection span stretched to **33/112 spans > 30 s (max 198 s)**. In this run **all 33 were chaos-rejected anyway** (0 contaminated vectors — the 10 real vectors all sit on ≤ 12 s spans), but a quiet-camera clip with a sparse track could yield a "reaction" vector measured over minutes of drift. Filed with a recommended span cap for selection.

## Ops note — crash + resume
The first attempt died silently overnight at 25/50 (the recurring macOS python hard-crash, no traceback). The per-video atomic writes + `force=False` resume worked exactly as designed: relaunch skipped the 25 finished clips and completed the rest (1,435 s). No data loss.

## Artifacts
`manifest_03d_50.json`, `03d_result_50.json`, `run_03d_50.log`, `frames/approach_*_343f4d2d.jpg`, `frames/avoidance_*_599f2f09.jpg`. Runner: `tools/run_03d_50.py` (now `force=False`).

## Verdict
- **03d is now functionally alive and honest**: real Approach/Avoidance vectors with verified sign semantics, SAM-masked depth medians on MPS, and honest chaos/sentinel rejection elsewhere.
- Yield (10 confident vectors / 50 clips) is modest but consistent with the corpus: most windows are dominated by wearer ego-motion — the documented modality limit, not a defect.
- One latent geometry risk (unbounded anchored span) is documented for selection; everything else from June 9 is resolved.
