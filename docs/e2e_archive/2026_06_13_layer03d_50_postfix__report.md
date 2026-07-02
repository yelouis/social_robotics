# Layer 03d — Identity-Collision Fix Validation (post-fix re-run)

**Date**: June 13, 2026 · **Host**: Mac Studio M4 Max, MPS · Same 50-clip manifest as the June 13 smell test (built by the *old* Node-02 fallback, so it still contains the collision — this run exercises the 03d-side guard end-to-end).

## What was fixed
Resolves docs/03d **Resolved #4** (identity collision) and **#5** (sentinel legibility), the two June 13 smell-test findings.

- **Issue 1 → Option A (upstream root cause)** — `social_presence.py`: untracked detections (`box.id is None`) now get a unique **negative** id instead of `len(frame_detections)`, which had collided with ByteTrack's positive track-id namespace (a child at t=21 inheriting the banner-holder's `person_id=2`). Validated on the exemplar: child → id −7, `person_id=2` reduces to the banner-holder `[t=0, t=9]`.
- **Issue 1 → Option C (downstream defense, repairs existing manifests)** — 03d identity-continuity guard: zeroes a vector (`identity_discontinuity: True` + `min_consecutive_iou`) when consecutive in-window boxes have IoU ≤ 0.1 **and** area ratio outside [0.6, 1.6]. Placed **after** the chaos gate so it targets only chaos-survivors (the confident false positives the gate misses) and preserves chaos attribution. Benign camera-pans (IoU ≈ 0 but area ratio ≈ 1.0) are spared.
- **Issue 2 → Option A** — provenance-only sentinel reasons; scoring path unchanged.

## Validation (this re-run vs the June 13 pre-fix run)
| metric | pre-fix (June 13) | post-fix |
|---|---|---|
| **Avoidance actions** | **1** (the false −0.48) | **0** ✅ |
| **Approach actions** | 1 (`343f4d2d` +0.41) | **1** (retained, min IoU 0.524) ✅ |
| identity_discontinuity rejections | — | **3** — all genuine collisions (`599f2f09` pid 1 & 2, `18323b66`) |
| clean non-zero vectors (real signal) | 13 | **13** (all preserved) |
| chaos-rejected | 60/79 | **60/79** (attribution unchanged) |
| scored / sentinels | 35 / 15 | 35 / 15 |
| sentinel reasons | all generic | **7 span-capped · 6 mixed · 2 single-detection** |

### Key cases (all correct)
- `599f2f09` pid2 (the −0.48 collision): → `identity_discontinuity`, vector 0.0 ✅
- `18323b66` pid0 (harmful switch, was conf 0): → `identity_discontinuity`, vector 0.0 ✅
- `343f4d2d` pid12 (**genuine** Approach +0.41): preserved, not flagged (min IoU 0.524) ✅
- `599f2f09` pid0 & `10167fcf` pid0 (benign camera pans): preserved, not flagged ✅

The headline false positive — the run's strongest Avoidance, a tracking-collision artifact — is gone, and the layer's single genuine action (the Approach) is untouched.

## Tests
+11 new (guard flags collision / spares pan / spares smooth track / single-box safe / end-to-end zeroing / negative-id fallback / 4× sentinel reasons / min_consecutive_iou provenance) and 2 window-contract assertions updated. Full suite **157 passed**.

## Notes
- The upstream fix (negative id) only benefits **newly-generated** manifests; the 03d guard is what repairs this and all previously-generated manifests at scoring time. A future full Node-02 re-run will let the upstream fix take effect corpus-wide.
- Flagged for later (not a blocker): bystander-anchoring measures some genuine vectors 40–70 s from the wearer's climax — a task-reaction-locality fidelity question recorded in docs/03d.

## Artifacts
`03d_result_50.json`, `run.log`, `supervise.log`, `analyze.py`, `manifest_03d_50.json`.
