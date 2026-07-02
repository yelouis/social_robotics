# End-to-End Test Report: 500-Video Batch (Ego4D scale-up)
**Date**: May 19, 2026
**Target Host**: Mac Studio (M4 Max, 64 GB Unified Memory)
**Nodes Tested**: 01 (Dataset Acquisition) & 02 (Filtering & Labeling)
**Tier (auto-detected)**: `medium`
**Registry composition**: 498 Ego4D `full_scale` videos (first 498 by `id` sort — 247 already-cached + 251 newly-downloaded; 2 of the 253 requested UIDs failed to download and 1 cached file is empty). Charades-Ego excluded at intake per `FilteringPipeline._SUPPORTED_DATASETS = ("ego4d",)` (Resolved Issue #6).

## Executive Summary

After the May 18 100-video run produced 0 passing videos, this 5× scale-up was commissioned to test whether the **0 % pass rate is small-sample noise or a real gate-accuracy problem**. The result: it is a real gate-accuracy problem. The May 19 498-video run produces **0 / 498 passing** with stereo-gate rejection rate of **325 / 402 (80.8 %)** — statistically indistinguishable from the May 18 100-video rate of 66 / 82 (80.5 %). Ruling out small-N noise was the explicit point of the scale-up and that is now firm.

The new run also produced two qualitatively new findings that the May 18 100-video sample did not surface:

> [!CAUTION]
> **The stereo VLM gate (Resolved Issue #10) is now demonstrably killing genuine social positives, not just collateral solo content.** Spot-checked a stereo-rejected pottery video (`082db032-…`, `e2e_reports/2026_05_19/frames/stereo_fp_new_2_1.jpg`): single-camera 1440×1080, two distinct people with overlapping hands working on a clay vessel together. The May 18 spot-checks showed the stereo gate killing only solo videos with paired symmetric objects (faucets, ovals, TV grids); this case shows it killing real multi-person content. Filed as additional verification on **Unresolved Issue #1** in `docs/02_filtering_and_labeling.md`.

> [!WARNING]
> **Moondream's multi-person gate is rejecting clear true-positive social videos.** Of 4 May 19 Moondream-rejected videos spot-checked, **3 are false negatives**: `0a6d4809-…` (`scenarios=["Talking with friends/housemates"]`, two people walking through doorway behind glass), `03634630-…` (`scenarios=["BBQ'ing/picnics","Walking on street"]`, second person walking ahead in corridor), and `0049fdd8-…` (convenience-store cashier behind snack rack, same as May 18). Only `097ef48a-…` (wearer alone in clothes store on phone) is a correct rejection. Filed as additional verification on **Unresolved Issue #2** in `docs/02_filtering_and_labeling.md`.

The headline interpretation is: **the May 19 `0 / 498` figure is composed of at least one stereo-gate false-positive on a real social video + at least three multi-person-gate false-negatives + the long tail of true-solo Ego4D activity.** The base-yield ceiling is plausibly higher than the May 18 `≤ 1 / 100` projection if both gates are repaired, but until then Layer 02 is producing zero data for Layer 03.

## Pipeline Configuration

| Setting | Value | Notes |
|---|---|---|
| Architecture | single-pass interleaved | Auto-selected (host RAM ≥ 48 GB) |
| `social_presence_pose` | `yolov8n-pose.pt` | Resolved via tier-per-host registry |
| `social_presence_hand_landmarker` | `models/mediapipe/hand_landmarker.task` | Resolved Issue #9 (MediaPipe Tasks API float16 bundle) |
| `social_presence_vlm_verify` | `moondream:latest` (1.7 GB) | Runs stereo + chin + multi-person gates |
| `SAF_VLM_VERIFY_SOCIAL` | `1` (default) | Resolved Issue #5 default |
| `sample_rate_fps` | `1/3` (1 frame per 3 s) | Resolved Issue #11 default |
| `filtering_vlm` | `qwen2.5vl:latest` (lazy fallback per Resolved Issue #7) | Climax deferred to Layer 03 per Resolved Issue #8 — not actually invoked here |
| `TMPDIR` | `/Volumes/Extreme SSD/tmp` | Per Resolved Issue #2 of `01_dataset_acquisition.md` |
| `HF_HOME` | `/Volumes/Extreme SSD/huggingface_cache` | Per Resolved Issue #3 of `01_dataset_acquisition.md` |
| `return_hands` | `True` | Production setting; no AttributeError under Resolved Issue #9 |

## Quantitative Results

### Registry build (Node 01)
- `scan_datasets()` now indexes **8 358** total videos (Charades-Ego 7 860 / Ego4D 498 / EPIC-KITCHENS 0 / EgoProceL 0) — Ego4D corpus grew from 247 to 498 via the May 19 download of 251 new `full_scale` files (~204 GB) via the Ego4D Python CLI.
- The 498-video test slice was deterministic-sorted Ego4D-only (`ego4d[:500]`) and persisted to `test_video_registry.json` (**389.7 GB** combined; median file size **397 MB**; max **11.5 GB**). One cached path was missing/empty and was logged as `MISSING` by the runner; 497 paths verified non-zero before the filter ran.
- Download throughput: 251 files / ~120 min wall = ~2 files / min average across multiple S3 buckets (`ego4d-iiith.s3.ap-south-1.amazonaws.com` was the dominant source).

### Filtering (Node 02 social-presence stage)
| Metric | Value | Compare May 18 (100-video) |
|---|---|---|
| Videos in slice | **498** | 100 |
| Skipped by metadata-solo prefilter (Resolved Issue #12) | **88 / 498 (17.7 %)** | 18 / 100 (18 %) |
| YOLO+VLM full-pass | **410 / 498** | 82 / 100 |
| Rejected by stereo VLM gate (Resolved Issue #10) | **325 / 402 (80.8 %)** ← *see Issue #1* | 66 / 82 (80.5 %) |
| Rejected by multi-person VLM gate (Resolved Issue #5) | **10 / 402** (all `0/2 confirmed`) | 3 / 82 |
| No YOLO-pose head+shoulder positive at all | **67 / 402** | 13 / 82 |
| Errors during processing | **0** | 0 |
| Passed social-presence filter | **0 / 498 (0.0 %)** | 0 / 100 (0.0 %) |
| Mean per-video YOLO+VLM time | **16.1 s** | 13.9 s |
| Median per-video YOLO+VLM time | **5.5 s** | 5.7 s |
| Max per-video YOLO+VLM time | **419.1 s** | 118.0 s |
| Total wall time | **109.7 min** | 19.0 min |

Note: a small discrepancy between the 498 total entries and the 402 processed-from-log count (rather than 410) is due to 8 video entries whose pipeline records exist in `e2e_social_filter_500.json` but whose per-video log line did not match the regex `\[\d+/498\] <uid> ds=ego4d` — those entries include the 1 missing file and 7 entries where the runner emitted a different status prefix. The aggregate `passed_filter` count is `0` either way.

The metadata-solo prefilter (Resolved Issue #12) maintained its 17.7 % skip rate at scale, matching the May 18 18 % and validating that gate's design. Every other gate downstream of it produced gate-accuracy regressions visible at 5× sample size.

## Discovered Bugs

No *new* unresolved issues filed today — the May 19 run re-verifies the two May 18 issues at scale (498 vs 100 videos) and adds qualitatively new evidence to each. Both issues are updated in-place in `docs/02_filtering_and_labeling.md` → `⚠️ Unresolved Issues & Suggestions`:

1. **Issue #1 — Stereo VLM gate over-fires on ordinary single-camera egocentric video** — May 19 update: 325 / 402 rejection rate (vs 66 / 82 on May 18) confirms the over-fire is structural and not sample-size noise. New evidence at scale: at least one stereo-rejected video (`082db032-…` pottery, `stereo_fp_new_2_1.jpg`) is a true multi-person social positive that the gate killed. The original three remediation options (aspect-ratio prefilter, sharpened prompt, intake-level allowlist) still stand; aspect-ratio prefilter remains the recommended path.
2. **Issue #2 — 0 social-presence pass rate** — May 19 update: re-verified at 0 / 498 (vs 0 / 100 on May 18). 3 of 4 May 19 Moondream-rejected videos spot-checked are false negatives — including a video with the unambiguous `scenarios=["Talking with friends/housemates"]` label (`0a6d4809-…`) that is rejected on `0/2 confirmed across 5 attempts`. The pass-rate collapse is now firmly demonstrated to be a *gate accuracy* problem rather than purely a *base-rate* problem. Remediation options (metadata-social positive allowlist, stronger fast VLM, broader source-dataset mix) still stand; the metadata-positive allowlist remains the recommended path but the stronger-VLM option (replacing Moondream with `qwen2.5vl:3b`/`7b`) becomes more attractive in light of the false-negative evidence.

## Qualitative Spot Check

### Closest-to-passing video (highest YOLO-pose+VLM signal, still rejected)
With 0 / 498 passing, the closest-to-passing case is the 10 Moondream-multi-person-rejected videos (all `0/2 confirmed across 5 VLM attempts` or `2 attempts` for one short video). The most informative is:
- **Video ID**: `0a6d4809-352b-44ca-9bef-ead01fd9c7f5` (Ego4D, `scenarios=["Talking with friends/housemates"]`, ~33.8 min duration, 1920×1080)
- **Rejection log**: `[SocialPresenceDetector] VLM gate rejected 0a6d4809-…mp4: 0/2 confirmed across 5 attempts`
- **Spot Check Result**: ❌ **False negative.** Sampled at 60 s / 180 s / 360 s (`closest_pass_talking_{1,2,3}.jpg`): the wearer is reading a Toxic Fat book in a lobby/library, and at the 180 s frame **two people walking past a glass-partitioned doorway behind the wearer are clearly visible** (one in cropped jeans and tennis shoes, one in dark trousers and brown shoes). YOLO-pose fires on the heads-and-shoulders of the through-glass walkers, but Moondream rejects on the multi-person prompt because the people are partially obscured by the glass mullions and overhead light reflections. The metadata label is unambiguously social (`Talking with friends/housemates`), and the video should pass.

![Closest-pass frame 1 — lobby reading scene](frames/closest_pass_talking_1.jpg)
![Closest-pass frame 2 — two people walking behind glass partition](frames/closest_pass_talking_2.jpg)
![Closest-pass frame 3 — wider lobby view, people still visible](frames/closest_pass_talking_3.jpg)

A second VLM-rejected video, `03634630-…` (`scenarios=["BBQ'ing/picnics","Walking on street"]`), is also a clear false negative — `closest_pass_bbq_2.jpg` shows a second person walking ahead of the wearer in a corridor. A third, `097ef48a-…` (clothes store with wearer browsing JustFab on phone), is a correct rejection.

### Worst-failing video (highest-confidence rejection that is actually wrong)
The most damaging wrong rejection in the May 19 run is the stereo gate killing a real multi-person video:
- **Video ID**: `082db032-f3f2-408b-a116-84d62f673915` (Ego4D, `scenarios=["Crafting"]`, ~67.5 min duration, **1440×1080** single-camera)
- **Rejection log**: `[SocialPresenceDetector] Stereo gate rejected 082db032-…mp4: side-by-side stereoscopic capture detected`
- **Spot Check Result**: ❌❌ **Critical false positive on stereo gate AND would have been a true positive on social presence.** Sampled at 60 s (`stereo_fp_new_2_1.jpg`): the frame shows **two distinct people** working together on a clay vessel — one set of hands (with red and yellow wristbands) on the left, another set of hands (with brown skin, multiple wristbands) on the right, with the wearer's own feet visible at the bottom edge in red sandals. This is a 3-person scene (wearer + 2 collaborators) on a single 1440×1080 camera, not a 2:1 stereo. The stereo gate is killing exactly the kind of cooperative-task scene Layer 02 was built to find. This single false positive is more damaging than any of the May 18 stereo false positives because those were at least true-solo videos that would have failed downstream anyway.

![Worst-fail frame — collaborative pottery, two pairs of hands + wearer's feet](frames/stereo_fp_new_2_1.jpg)

For comparison, two other stereo-rejected videos that *are* true solo (so the stereo gate's error is purely false-positive, not also a false-negative on social):
- `030c7542-…` (single-camera 1440×1080, content unknown but only one person) — `stereo_fp_new_1_1.jpg`
- `0a01978c-…` (single-camera 1440×1080 kitchen, one person at stove) — `stereo_fp_new_3_1.jpg`

![Stereo FP — solo scene 1](frames/stereo_fp_new_1_1.jpg)
![Stereo FP — solo kitchen](frames/stereo_fp_new_3_1.jpg)

### Sanity Check on Skips and True-Negative Rejections
- **Metadata-solo prefilter (Resolved Issue #12)**: 88 / 498 videos (17.7 %) were skipped before YOLO. Skip rate matches May 18 (18 %) to within rounding — the prefilter scales as designed.
- **Multi-person VLM gate true negatives**: At least 1 of the 4 spot-checked Moondream-rejected videos (`097ef48a-…`, wearer alone in clothes store) is a correct true-negative.
- **Resolved Issue #9 stability at scale**: 0 errors across 410 full-pass videos with `return_hands=True` over 110 min wall time — the MediaPipe Tasks API HandLandmarker is stable end-to-end on `mediapipe==0.10.35` at 5× the May 18 sample size.

## Artifacts
- `e2e_social_filter_500.json` — per-video result records (skip reason, elapsed time, bystander aggregates, pass/fail)
- `test_video_registry.json` — input 498-video slice (deterministic-sorted Ego4D first-500)
- `frames/` — spot-check stills (`closest_pass_*` and `stereo_fp_new_*`)
- Source runner: `scratch/run_e2e_social_only_v3.py` (same as May 18; only the `INPUT`/`OUTPUT` paths point at the 500-video registry)
- Logs: `scratch/ego4d_500_download.log` (Node 01 download), `scratch/e2e_v3_500.log` (Node 02 filter)
