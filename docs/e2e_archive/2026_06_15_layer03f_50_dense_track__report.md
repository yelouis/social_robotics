# Layer 03f — Dense-Track Gate (Issue 1 Option A) Validation

**Date**: June 15, 2026 · **Host**: Mac Studio M4 Max, MPS · Same 50-clip manifest. 239 s, 0 crashes.

## Headline: motor_resonance 2 → 0 (the camera-motion-through-bbox FPs are gone)
| metric | gated (Resolved #9) | dense-track (Option A) |
|---|---|---|
| scored | 34 | 34 (unchanged) |
| **motor_resonance_detected** | 2 | **0** |
| mirroring_detected | 6 | 6 (not gated — separate metric) |
| velocity_peak | median 4.65 | median 4.18 (sparse rows now 0.0, honest) |
| empathy_scalar | 2 non-zero | 0.0 (gated behind resonance) |

The gate (`_has_dense_track`, `MIN_GENUINE_DETECTIONS = 2`) computes pose velocity only when ≥ 2 **genuine** Node-02 detections fall inside the measurement window `[w_start, w_end]`. A single carried bbox (or a track whose only in-window detection is one) is effectively a fixed/drifting box: a panning camera drags the scene through it and YOLO-pose reports apparent keypoint velocity time-locked to the jolt **by construction**. Sparser tracks now emit velocity 0 / no resonance (honest). Mirroring (spine-angle correlation) is a separate signal and is left untouched.

## Per-row cross-check vs the gated run — no collateral damage ✅
Joining the two runs by `(video_id, person_id)` (72 rows each):
- **9 rows changed, every one a single-in-window-detection track** (`genuine_in_window = 1 < 2`); **0 rows needed review.**
- **63 rows with non-zero velocity are bit-identical** (rounded) to the gated run — every genuine dense track (≥ 2 in-window detections) is untouched.
- The change is exactly "zero the sparse-track velocities, keep the dense ones."

| video/pid | gated vel | new vel | gated res | new res | genuine_in_window |
|---|---|---|---|---|---|
| `0235dafb` pid0  | 3.42 | 0.0 | **True** | False | 1 |
| `66d4121f` pid16 | 5.37 | 0.0 | **True** | False | 1 |
| `0c163d16` pid21 | 8.76 | 0.0 | False | False | 1 |
| `0c163d16` pid22 | 8.69 | 0.0 | False | False | 1 |
| `630bd4ba` pid0  | 4.65 | 0.0 | False | False | 1 |
| `10167fcf` pid0  | 4.61 | 0.0 | False | False | 1 |
| `48e621af` pid6  | 0.85 | 0.0 | False | False | 1 |
| `044a7a23` pid4  | 0.70 | 0.0 | False | False | 1 |
| `044a7a23` pid3  | 0.47 | 0.0 | False | False | 1 |

## Spot-check — the 2 survivors of Resolved #9 are now rejected ✅
Both were visually confirmed as fixed/sparse-box artifacts in the gated run (`../2026_06_15_layer03f_50_gated/frames/{66d4121f,0235dafb}_*_rw.jpg` — the box content morphs as the wearer's camera pans). The gate kills them deterministically:
- `66d4121f` pid16 (1 detection @ 33.0 s): its scoring task (rw 32.70–34.70) has `genuine_in_window = 1` → velocity 0, resonance False.
- `0235dafb` pid0 (6 detections, 12 s gaps): its scoring task (rw 2.07–4.07) has `genuine_in_window = 1` → velocity 0, resonance False (the other task was already `span_capped`).

An offline replay of the window helper + gate against the manifest predicted this before the GPU run (both survivors gated on every task); the run confirmed it.

## Verdict
- **Option A resolves the camera-motion-through-bbox FP class**: motor_resonance 2 → 0, scored intact (34), mirroring intact (6), and every dense track's velocity is preserved exactly. → docs/03f **Resolved #10**. Suite 195/195 (+ `test_has_dense_track_gates_single_and_sparse`).
- *Cons realized as documented*: yield drops — 9 single-detection velocities (incl. 2 that had fired resonance) become honest 0.0. This is the selected trade-off ("no real track → no claim"). Option B (ego-motion compensation) remains available later if sparse-but-multi tracks need rescuing.

## Artifacts
`03f_result_50.json`, `run.log`, `supervise.log`, `analyze.py`, `manifest_03f_50.json`. Survivor frames live in the gated run dir (unchanged).
