# Layer 03b (Reasonable Emotion) — Functional Validation & Spot-Check Report

**Date**: June 4, 2026
**Goal**: Confirm Layer 03b runs end-to-end on real Node-02 output and that its emotion-derived success/failure scores pass a visual smell test (does the detected bystander emotion — and the resulting positive/negative direction — match the actual face?). Correctness probe, not a benchmark.
**Host**: Mac Studio (M4 Max, 64 GB), MPS.

## Terminology
"Filtered clips" = videos retained in `filtered_manifest.json` (passed Node 02). 03b scores each *task* in a clip via `task_aggregate_score ∈ [−1, +1]` (attention-weighted mean of per-person `late_stage_weighted_success_score`; **+** = bystanders reacted as if the task succeeded, **−** = as if it failed).

## Sample
10 filtered clips bounded to `duration ≤ 300 s` (so the climax optical-flow pass is tractable), sampled across the Node-02 score range (1.2 → 49.8). The dog clip is excluded (flagged).

## Getting 03b to run at all — four blockers hit before any scoring
03b is **not runnable end-to-end as shipped**. To produce a real result I had to work around:
1. **No climax metadata.** 03b reads `task_temporal_metadata.task_reaction_window_sec` and returns `None` if empty (pipeline.py:374–378), but the manifest ships with empty climax metadata and **03b never calls `populate_climax_for_manifest`** (the documented "first Layer 03 populates it" contract, 02 Resolved Issue #8, is unmet). → Runner calls it first; the optical-flow pass cost **988 s for 10 short clips** (~99 s/clip; far worse on full-length clips).
2. **HSEmotion (the emotion model) will not load** under this environment — three compounding breaks: torch ≥ 2.6 `weights_only=True` default, a **CUDA-pickled** checkpoint (`map_location` needed on this non-CUDA Mac), and a **timm-version pickle mismatch** (`DepthwiseSeparableConv has no attribute 'conv_s2d'` on timm 1.x; `timm.layers` missing on timm 0.6.x). So `emotion_detector` is `None`.
3. **Silent MOCK fallback.** With HSEmotion `None`, `_sample_emotions` silently substitutes a **seeded-RNG mock emotion distribution** (pipeline.py:600–611) — 03b produces fake-emotion-driven scores with no loud warning. *This is the default behavior on a clean install.* I injected the **ONNX** HSEmotion backend (`hsemotion-onnx`, real weights, drop-in `predict_emotions`) so this report uses **real** emotions.
4. (Run was performed with `pipeline.py` **unchanged** — these are documented for your decision.)

## Headline result: 03b does not produce meaningful output on real Ego4D data
- **2 / 10 clips scored. 8 / 10 failed to score.**
- Both scored clips returned **0.04** (near-zero), and inspection shows the underlying emotions are **classifier noise**, not real reactions.

### Scored clips
| clip | task | window (s) | `task_aggregate_score` | persons |
|---|---|---|---|---|
| `630bd4ba` | Hiking | 122.0–124.0 | **0.04** | 1 |
| `10167fcf` | Indoor Navigation | 16.8–18.8 | **0.04** | 1 |

### Failed-to-score (8/10) — all the same cause
| clip | reaction window (s) | climax (s) | nearest bystander detection |
|---|---|---|---|
| `6b3988dd` | 115–117 | 114 | **28 s away** |
| `2d14ed1e` | 22–24 | 21 | 9 s away |
| `341b5211` | 25–27 | 24 | 30 s away |
| `28539222` | 46–48 | 45 | 22 s away |
| `14de41ea` | 46–48 | 45 | 15 s away |
| `2c6e772c` | 35–37 | 34 | 26 s away |
| `66fa8650` | 107–109 | 106 | 110 s away |
| `573fc64b` | 14–16 | 13 | 8 s away |

**Every failure has no bystander detected within the reaction window.** 03b samples emotions in the window, finds no bystander (the `>2.0 s` skip drops every sample), gets <2 samples, and returns `None`.

## Spot checks

### Highest (and tie): `630bd4ba` "Hiking", score 0.04 — ❌ no face, noise
![hiking — backpack, no face](frames/sc03b_scored_hiking_3.jpg)
The bystander box is on a seated hiker's **back/backpack — no face is visible**. ONNX HSEmotion labeled it "Happiness" at magnitude **0.19** (≈ the 1/8 = 0.125 uniform baseline → the model is guessing). The emotion sequence over 2 s was neutral→joy→neutral; the 0.04 score is noise on a faceless crop.

### Lowest (and tie): `10167fcf` "Indoor Navigation", score 0.04 — ❌ distant full-body, fixed camera, noise
![nav — distant full body](frames/sc03b_scored_nav_2.jpg)
The same fixed-camera room clip flagged during the 03a check. The box is a **distant full-body** man at a desk — not a face crop. Emotions flip neutral→**fear**→**joy**→**surprise**→**disgust** in under 2 s (all magnitude ~0.16–0.25): physically impossible for a real face, classic classifier noise on a non-face crop.

### Failed-to-score: `6b3988dd` — bystander present at t=87 s, reaction window at t=116 s
![bystander actually present at 87s](frames/sc03b_failed_bystander_actual.jpg)
![reaction window at 116s — empty street](frames/sc03b_failed_window_NObystander.jpg)
A pedestrian (boxed) is detected at **t=87 s**; the optical-flow reaction window is at **t=116 s**, by which point the wearer has walked on and the street is **empty**. No bystander in the window → no score. This temporal decoupling — the optical-flow climax tracks the **wearer's** kinetic peak, not bystander presence — is the dominant failure mode (8/10).

## Smell-test verdict
**03b runs, but its output is not yet meaningful on real egocentric data.** Two structural problems compound:
1. **Window ↔ bystander mismatch** (8/10 score nothing): the reaction window is anchored to the wearer's optical-flow climax, which is usually not when a bystander is on-camera.
2. **Non-face crops → emotion noise** (the 2 that score): Node-02 bystander boxes are full-body/partial/back-of-person; the face-emotion model returns near-uniform (~0.18) guesses, so the success scores are noise.
Plus the three infra blockers (climax not auto-populated, HSEmotion won't load, silent mock fallback). All are filed below for your decision.

## Issues filed (per the Bug Documentation Style Guide) → `docs/03b_reasonable_emotion_layer.md`
- **Issue 2**: Reaction window decoupled from bystander presence → 8/10 clips score nothing.
- **Issue 3**: Full-body / non-face crops fed to HSEmotion → near-uniform-confidence emotion noise (no face gate; magnitudes ignored).
- **Issue 4**: HSEmotion torch backend fails to load (torch ≥2.6 / CUDA-pickle / timm) → silent mock-emotion fallback.
- **Issue 5**: 03b never populates climax metadata → 0 results on the shipped manifest (unmet Layer-03 contract).

## Artifacts
- `manifest_03b_10.json` (climax-populated), `03b_result_10.json`
- `frames/sc03b_scored_*` (the 2 scored, faceless/noise), `frames/sc03b_failed_*` (window vs bystander timing)
- Runner: `scratch/run_03b_10.py` (ONNX-backend injection); spot-check: `scratch/spotcheck_03b.py`; log: `scratch/run_03b_10.log`
