# Layer 03f — Resonance FP Gate (Issue 1 Option A) Validation

**Date**: June 15, 2026 · **Host**: Mac Studio M4 Max, MPS · Same 50-clip manifest. 242 s, 0 crashes.

## Headline: motor_resonance 9 → 2 (false positives cut)
| metric | pre-gate (dense) | gated (Option A) |
|---|---|---|
| scored | 34 | 34 (unchanged) |
| **motor_resonance_detected** | 9 | **2** |
| mirroring_detected | 6 | 6 (not gated — separate metric) |
| velocity_peak | median 4.65 | median 4.65 (unchanged) |

The gate (`_resonance_decision`) requires (a) an **impulsive** flinch — `max_vel ≥ VELOCITY_NORMALIZER` AND `≥ RESONANCE_IMPULSE_RATIO (3.0) × the bystander's own median velocity` (sustained eating/walking is high-but-flat → ratio ~1 → rejected) — and (b) correlation with only the single **dominant** ego spike (max chaos score), not the whole relative-threshold list.

## Spot-check — the documented FPs are now rejected ✅
- `43bd06f3` pid0 (the **eating** man) → `resonance=False` ✓
- `599f2f09` pid0 (the **festival crowd**) → `resonance=False` ✓

Both were the exemplars cited in the Issue; the impulse + dominant-jolt gate removes them.

## Spot-check — the 2 survivors are a *new, subtler* FP class → new Issue 1
Re-rendered over their correct (in-window) reaction windows:
- `66d4121f` pid16 (**1 detection**, rw [32.7,34.7]): the fixed bbox's content morphs from a person-in-black to a yellow-vest worker as the **wearer's camera pans** — apparent "pose velocity" is the scene moving through a static box, time-locked to the jolt by construction.
- `0235dafb` pid0 (6 detections, median 12 s gaps, rw [2.07,4.07]): sparse track, box drifts over background.

→ Camera motion leaks *into* the bystander pose velocity on single/sparse tracks. Distinct from Resolved #7 (which masked the bystander *out of* ego chaos). Filed as docs/03f **Unresolved Issue 1** with options (dense-track gate / ego-motion compensation / track-density provenance).

## Verdict
- **Option A resolves the documented FP classes** (sustained motion + spike-abundance coincidence): 9 → 2, eating/crowd rejected, scoring intact. → docs/03f **Resolved #9**. Suite 195/195 (+ gate unit test).
- A residual, subtler FP class (camera-motion-through-fixed-bbox) is newly surfaced and documented for selection — the same iterative pattern as the rest of the 03 stack.

## Artifacts
`03f_result_50.json`, `run.log`, `analyze.py`, `frames/{43bd06f3,599f2f09}_pid*_strip.jpg` (rejected FPs), `frames/{66d4121f,0235dafb}_*_rw.jpg` (survivors over correct windows).
