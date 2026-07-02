# End-to-End Test Report: 150-Video Run (post gate-accuracy fixes + Layer 1a)
**Date**: May 22–23, 2026
**Target Host**: Mac Studio (M4 Max, 64 GB Unified Memory)
**Nodes Tested**: 01 (Dataset Acquisition) · 02 (Filtering & Labeling) · 01a (Synthetic True-Positive Generation)
**Tier (auto-detected)**: `medium` (host ≥ 48 GB → `qwen2.5vl` multi-person gate, resolved to `qwen2.5vl:latest`)
**Registry composition**: 150 Ego4D `full_scale` videos, deterministically **stride-sampled** (every ~3.32nd entry) from the 498-video id-sorted slice for spread across the corpus (160.7 GB; median 491 MB; max 12.37 GB). **0 synthetic_validation videos** — Layer 1a generation could not produce a clip on this host (see § Synthetic-positive verification).

> [!NOTE]
> **Scope**: The user requested a 500-video run; after measuring the per-video cost of the new `qwen2.5vl` gate (~141 s/processed video, ~5–8× the May 19 Moondream pace), we agreed on a **representative 150-video slice** (~5 h) to get a statistically meaningful pass rate plus the full spot-check and Layer 1a validation in one night rather than a ~9–12 h full pass.

## Executive Summary

This run measures the two May 22 gate-accuracy fixes against the May 19 baseline of **0 / 498 passing**. The result is a decisive improvement: **43 / 128 processed videos pass (33.6 %)**, and the two failure modes that produced the May 19 zero are both addressed:

- **The stereo-gate over-fire is eliminated.** May 19 rejected **325 / 402** videos as "side-by-side stereo." This run: **0** stereo rejections. Resolved Issue #13's aspect-ratio prefilter (skip the stereo VLM when `width/height < 1.9`) means normal 4:3/16:9 egocentric video never reaches the stereo prompt.
- **The multi-person gate now recovers true positives.** Swapping Moondream → `qwen2.5vl` (Resolved Issue #14) recovered every May 19 false-negative we re-checked in the 8-video smoke (`0a6d4809` "Talking with friends", `03634630` "BBQ", `0049fdd8` convenience-store cashier — all now pass), and the run's strongest pass is an unambiguous 2.3-hour city-plaza walk teeming with bystanders.

> [!IMPORTANT]
> **Layer 1a (synthetic true positives) could not be exercised: generation is blocked on this host.** Across four render attempts no clip completed end-to-end — the Wan2.1 generator hits overlapping memory walls on the 64 GB MPS backend (14B attention-buffer OOM; 1.3B CPU-VAE-decode SIGKILL at every filter-viable clip length). This is documented in `docs/01a_synthetic_positive_generation.md` → Unresolved Issues #1 and #2. **Consequently the "does 02 catch the 01a true positives?" check was answered against real Ego4D true positives instead of synthetic fixtures** — and 02 does catch them (best-pass below + the three recovered smoke false-negatives). The `SAF_RUN_SYNTHETIC_QA` toggle and the generate-once/reference-thereafter plumbing are in place, so the synthetic QA loop closes as soon as the generator decode is fixed.

The headline is **not** "the filter is now correct" — it is "the filter went from unusable (0 %) to usefully selective (33.6 %), with two smaller, well-characterized residual error classes" (a multi-person **false-negative** class on long/mixed videos where the bystander is intermittently framed, and a **false-positive** class where `qwen2.5vl` over-confirms on odd single-person framing). Both are filed in `docs/02_filtering_and_labeling.md` → Unresolved Issues.

## Pipeline Configuration

| Setting | Value | Notes |
|---|---|---|
| Architecture | single-pass interleaved | host RAM ≥ 48 GB |
| `social_presence_pose` | `yolov8n-pose.pt` | head+shoulder keypoint gate |
| `social_presence_vlm_verify` | **`qwen2.5vl:7b` → `qwen2.5vl:latest`** | Resolved Issue #14; `:7b` not installed, lazy tag resolver fell back to `:latest` (Resolved Issue #7) |
| Multi-person gate | `qwen2.5vl` YES/NO, `min_consistency=2`, `MAX_VLM_VERIFY_FRAMES=5` | |
| Stereo gate | aspect-ratio prefilter (`w/h ≥ 1.9`) → VLM | Resolved Issue #13 — never fired this run |
| Wearer-chin gate | bottom-20 % bbox → `qwen2.5vl` upward-face check | Resolved Issue #11 |
| `sample_rate_fps` | `1/3` | Resolved Issue #11 |
| `SAF_VLM_VERIFY_SOCIAL` | `1` (on) | |
| `SAF_RUN_SYNTHETIC_QA` | `1` (on, default) | **new toggle**; no synthetic clips existed so it was a no-op this run |
| `synthetic_generator` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` (medium) | OOM on MPS; 1.3B fallback also blocked at decode |
| Filter env | `venv` (Python 3.9.6, psutil → `medium`) | system `python3` mis-detects `small`; see Observations |

## Quantitative Results

### Node 01 (Dataset Acquisition)
- Input registry: the existing 498-video Ego4D `full_scale` corpus on the Extreme SSD; this run used a deterministic **150-video stride subset** (`e2e_reports/2026_05_22/registry_150_ego4d.json`).
- Layer 1a stream: **0 videos** registered — generation produced no clip (see below).

### Node 02 (Filtering & social-presence stage)
| Metric | This run (150) | May 19 (498) |
|---|---|---|
| Videos in slice | 150 | 498 |
| Skipped by metadata-solo prefilter | 22 / 150 (14.7 %) | 88 / 498 (17.7 %) |
| Processed (YOLO+VLM) | 128 | 402 |
| **Passed social-presence filter** | **43 / 128 (33.6 %)** | **0 / 498 (0.0 %)** |
| Rejected by **stereo** gate | **0** | 325 / 402 (80.8 %) |
| Rejected by **multi-person** gate | 62 | 10 |
| No YOLO-pose positive / all-filtered | 23 | 67 |
| Errors | 0 | 0 |
| Mean per-video YOLO+VLM time | 141.2 s | 16.1 s |
| Median per-video time | 84.1 s | 5.5 s |
| Max per-video time | 3603.6 s (2.3 h video) | 419.1 s |
| Total wall time | 301.2 min | 109.7 min |

**Pass-quality distribution** (social_presence_score = summed detection confidence across sampled frames; scales with length × crowd density): 20 strong (≥ 50), 14 mid (10–50), **9 weak (< 10)**; median score 45.4, median bystander count 9. The 9 weak passes are the false-positive-risk band (see spot checks).

The per-video time rose ~9× vs May 19 because `qwen2.5vl` runs at ~52 GB / 128k context (vs Moondream's 1.7 GB) and because the ~325 videos that previously short-circuited on the stereo gate (one Moondream call) now fall through to the full multi-person gate (up to 5 `qwen2.5vl` calls).

## Synthetic-positive verification (Layer 1a)

**Status: blocked — no synthetic fixtures could be generated on this host.** The intended per-scenario pass-rate table is therefore empty:

| Scenario tag | Generated | Passed Layer 02 | Pass rate |
|---|---|---|---|
| handoff / flinch / nod_smile / gaze_check | **0** | — | — |

Four render attempts, all on the validated-only-on-smoke local Wan2.1 path:

| Attempt | Config | Outcome |
|---|---|---|
| 1 | 14B / 720p / 17f (smoke) | SIGABRT — `Failed to allocate private MTLBuffer for size 51840000000` (~48 GiB) at denoise step 0 |
| 2 | 1.3B / 480p / 81f | same ~48 GiB MTLBuffer OOM at denoise step 0 |
| 3 | 1.3B / 480p / 49f | SIGKILL 137 at end of denoise / start of decode |
| 4 | 1.3B / 480p / 33f | completed denoise; **SIGKILL 137 during CPU VAE decode** |

Root cause is two overlapping memory walls on the 64 GB MPS backend: (a) the Wan DiT materializes the full O(seq²) self-attention score matrix (no memory-efficient attention kernel on MPS), so ≥ 81 frames or 720p OOMs the Metal buffer; (b) the CPU VAE decode exhausts unified memory at every clip length long enough to clear Layer 02's `min_consistency=2` gate (≥ 49 frames at 1/3-fps sampling). Resolved Issue #3 ("CPU decode OOM fixed") was only ever validated on a 17-frame smoke and does not hold at real lengths. Full write-up + remediation options (temporal-chunked decode is the recommended fix) are in `docs/01a` → Unresolved Issues #1 and #2.

**True-positive catching was instead validated on real Ego4D data** (which is the property the synthetic stream is a proxy for): the best-pass below, plus the three smoke false-negatives `qwen2.5vl` recovered (`0a6d4809`, `03634630`, `0049fdd8`). The new `SAF_RUN_SYNTHETIC_QA` toggle and the generate-once → save-to-SSD → reference-thereafter workflow are implemented and documented, so once the generator decode is fixed, a single isolated generation run closes the loop with no further pipeline changes.

## Discovered Issues

Documented per the Bug Documentation Style Guide:

- **`docs/01a` Unresolved #1** — Wan2.1-14B generation OOMs at denoise step 0 on the 64 GB MPS host (attention buffer); 14B was never live-validated.
- **`docs/01a` Unresolved #2** — CPU VAE decode SIGKILL-137s at every filter-viable clip length; Resolved Issue #3 is a false resolution (validated only on a 17-frame smoke).
- **`docs/02` Unresolved (new)** — Residual multi-person **false-negative** on long/mixed videos with intermittently-framed bystanders (worst-fail `025c24db` below): the `MAX_VLM_VERIFY_FRAMES=5` × `min_consistency=2` × 1/3-fps sampling can miss a genuine but minority social segment.
- **`docs/02` Unresolved (new)** — Multi-person **false-positive** where `qwen2.5vl` over-confirms on single-person/non-egocentric framing (smoke video `0a01978c`, a fixed-camera single occupant; and the weak-score pass band).

No new errors/crashes in Node 02 (0 / 128).

## Qualitative Spot Check

All spot-checks were performed by opening the actual frames (faces are dataset-privacy-blurred).

### Video that passed the filter the BEST — `0752c643-18c8-4fd3-9a32-7ec985f2a6bd`
- score **3464.4**, 382 tracked bystanders, 4170 frame-detections, 2.3-hour egocentric video, 1920×1080.
- **Spot Check: ✅ Strong true positive.** A first-person walk through a Latin-American city (Spanish signage "Bienvenidos Catedral…", colonial plazas). Sampled at 15 s / 2127 s / 6380 s: a companion beside the wearer in a cathedral plaza (boxed), three friends walking together past street vendors, and a face-to-face conversation with a masked companion in a stone plaza. Genuinely, continuously social. The enormous score is partly length-driven (summed confidence over a 2.3 h clip), but the per-frame content is unambiguously multi-person — 02 passed it correctly and confidently.

![Best pass — cathedral plaza, companion boxed](frames/sc150_bestpass_0752c643_1.jpg)
![Best pass — friends walking past street vendors](frames/sc150_bestpass_spread_0752c643_1.jpg)
![Best pass — face-to-face conversation in plaza](frames/sc150_bestpass_spread_0752c643_3.jpg)

### Video that FAILED the filter the WORST — `025c24db-1ab3-462e-888a-3598fd777ee2`
- Rejected: `VLM gate rejected … 1/2 confirmed across 5 attempts`, ~53-min video, 1440×1080.
- **Spot Check: ❌ False negative.** Sampled at 1067 s / 2133 s / 3200 s: the first two frames are an unambiguous **2–3 person Rummikub game** — a woman seated across the table, a third person's arm at the left edge, tiles spread out, the wearer's hands. The third frame is the same wearer **later, alone, reading on a couch**. The video genuinely contains a multi-person interaction, but it is a *minority* of a long mixed clip; sampling at 1/3 fps with only 5 VLM attempts and a 2-confirmation threshold, the gate confirmed multiple people only once and dropped the whole video. This is the most damaging wrong decision in the run — a clear social interaction lost.

![Worst fail — multi-person Rummikub game](frames/sc150_fn_025c24db_2.jpg)
![Worst fail — wearer's hand + opponent across table](frames/sc150_fn_025c24db_1.jpg)
![Worst fail — same video later, wearer alone on couch](frames/sc150_fn_025c24db_3.jpg)

### Re-checks that refine prior claims
- **`082db032` (pottery)** — May 19 called this a clear stereo-gate false-positive on a multi-person scene. Re-check: it is a *collaborative* clay-vessel scene, but the second person is only ever visible as **disembodied hands/arms reaching in** — never a head/torso. The new stereo gate correctly does **not** fire (AR 1.33). The multi-person gate's 1/2 rejection is **defensible**, not a clear false-negative: by the filter's own head+shoulder definition, hands-only is not a confirmable bystander. May 19's "clear FN" claim was overstated.
- **`0a01978c` (smoke, kitchen)** — ❌ **False positive.** A fixed wide-angle room camera (not egocentric) with a **single** occupant at a table; `qwen2.5vl` confirmed "multiple people" and passed it (low score 8.5). The clearest example of the FP class.
- **`0c190d90` (weak pass, score 4.23)** — ✅ correct-but-weak: a real but **distant, incidental pedestrian** on a winter street. Low scores are not automatically false positives.
- **`03d6e9bf` (1/2 reject)** — ✅ correct rejection: a dim **solo kitchen**; the single "yes" was VLM noise.

### Sanity check on skips
- Metadata-solo prefilter: 22 / 150 (14.7 %) skipped before YOLO — consistent with May 19's 17.7 % (the stride sample drew slightly fewer all-solo-scenario videos).

## Observations / Operational Notes
- **Tier detection depends on the environment.** The system `python3` lacks `psutil` → mis-detects `small` → would silently use Moondream (the May-19 config). The run must use `venv` (psutil present → `medium` → `qwen2.5vl`). Worth pinning `SR_MODEL_TIER=medium` for E2E runs to remove the ambiguity.
- `qwen2.5vl:7b` is not installed; the lazy tag resolver (Resolved Issue #7) substitutes `qwen2.5vl:latest`. Functionally correct, but the resident set is ~52 GB at 128k context — the dominant per-video cost and the reason generation cannot co-reside with the filter.

## Artifacts
- `e2e_reports/2026_05_22/e2e_social_filter_150.json` — per-video result records.
- `e2e_reports/2026_05_22/registry_150_ego4d.json` — the 150-video input slice.
- `e2e_reports/2026_05_22/frames/` — spot-check stills (`sc150_*`, `sc_pottery_*`, `sc_kitchen_*`).
- Runner: `scratch/run_e2e_social_only_v4.py` (synthetic-aware, `--input/--output/--limit`).
- Logs: `scratch/e2e_v4_150.log` (filter); `scratch/gen_*.log` (generation attempts).
- Analyzer: `scratch/analyze_e2e.py`.
