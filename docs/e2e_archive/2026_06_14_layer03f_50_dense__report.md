# Layer 03f — Dense Pose Sampling (Issue 1 Option A) E2E Validation

**Date**: June 15, 2026 · **Host**: Mac Studio M4 Max, MPS · Same 50-clip manifest. Runner: `run.py` (YOLOv8x-pose, `TARGET_POSE_FPS=10`). 299 s, 0 crashes, no error log.

## Headline: 0/50 → producing (the mechanical fix works E2E)
| metric | sparse (June 14) | dense (Option A) |
|---|---|---|
| scored | 6 | **34** |
| sentinels (`no_pose_data`) | 44 | 16 |
| `bystander_pose_velocity_peak` | 0.0 everywhere | min 0.34 / median **4.65** / max 10 |
| motor_resonance_detected | 0 | 9 |
| mirroring_detected | 0 | 6 |

The fix: `_extract_and_correlate_pose` anchors its window via the shared `bystander_measurement_window` helper, then samples YOLO-pose **densely** at `TARGET_POSE_FPS` across it (sequential grab/read, bbox interpolated between sparse detections via `_interp_bbox`). Velocity is now computable wherever the bystander is visible near the jolt — the ≥2-frame starvation is gone.

## Spot-check: the resonances are largely FALSE POSITIVES (filed as new Issue 1)
Two top detections, both **not** sympathetic flinches:
- **`43bd06f3` pid0** (vel 4.99, resonance, empathy 1.0): a man **eating** — hand-to-mouth motion (`frames/43bd06f3_pid0_strip.jpg`).
- **`599f2f09` pid0** (vel 7.55, resonance, empathy 1.0): a **festival crowd**; the bbox jumps between bodies (`frames/599f2f09_pid0_strip.jpg`).

Both peaks land within `RESONANCE_WINDOW_SEC` (0.5 s) of one of the constantly-firing ego spikes (`ego_kinetic_chaos_score` 0.97–1.0). Root: dense sampling captures *all* motion (eating/walking/crowd), the egocentric camera spikes almost constantly (multiple-comparisons coincidence), and the ego window (reaction window) and pose window (bystander-anchored) can be temporally offset. 9/34 resonance is implausibly high for genuine flinches.

## Verdict
- **Issue 1 (velocity starvation) is resolved E2E** — Option A does exactly what it promised: velocity computes, yield 0 → 34. → docs/03f **Resolved #8**.
- **Spot-check surfaced a new correctness problem**: `motor_resonance_detected` is now false-positive-prone (coincidental motion near abundant spikes). Filed as docs/03f **Unresolved Issue 1** with options (dominant-jolt + baseline-relative impulsive flinch / ego-pose temporal coherence / demote to provenance-tagged candidate). This mirrors the project pattern: fixing the mechanical starvation reveals the next-level quality issue.

## Artifacts
`03f_result_50.json`, `run.log`, `analyze.py`, `frames/43bd06f3_pid0_strip.jpg`, `frames/599f2f09_pid0_strip.jpg`.
