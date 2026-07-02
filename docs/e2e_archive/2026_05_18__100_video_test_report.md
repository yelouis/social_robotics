# End-to-End Test Report: 100-Video Batch
**Date**: May 18, 2026
**Target Host**: Mac Studio (M4 Max, 64 GB Unified Memory)
**Nodes Tested**: 01 (Dataset Acquisition) & 02 (Filtering & Labeling)
**Tier (auto-detected)**: `medium`
**Registry composition**: 100 Ego4D `full_scale` videos (first 100 by `id` sort) — Charades-Ego excluded at intake per `FilteringPipeline._SUPPORTED_DATASETS = ("ego4d",)` (Resolved Issue #6).

## Executive Summary

The four May 17 unresolved issues are resolved in `2832570` and the new defaults were exercised end-to-end: the MediaPipe Tasks API `HandLandmarker` replaces the removed `mp.solutions.hands` namespace (Resolved Issue #9), the stereo and wearer-chin VLM gates are wired into `SocialPresenceDetector` (Resolved Issues #10, #11), the social-presence sample rate dropped from 1 FPS to 1 frame per 3 seconds, and an Ego4D `scenarios`-based metadata prefilter skips obvious-solo videos before YOLO (Resolved Issue #12). The pipeline ran to completion on all 100 videos with **0 errors** — Issue #9's `AttributeError` storm is gone — and the metadata prefilter skipped 18 of 100 entries before paying any YOLO+VLM cost. Total wall-time dropped from **180.8 min (May 17)** to **19.0 min (May 18)** — a 9.5× speedup driven mainly by the 1/3 FPS sample rate and the metadata prefilter.

The run surfaced **two new unresolved issues**, both critical filter-quality regressions confirmed via spot-check:

> [!CAUTION]
> **The new stereo VLM gate (Resolved Issue #10) over-fires on 66 of 82 YOLO+VLM-processed videos** — including ordinary single-camera GoPro captures with no stereo split. Spot-checked four stereo-rejected videos (`stereo_fp_hedge`, `stereo_fp_hotel`, `extra_stereo_000786a7`, `extra_stereo_001e3e4e`); none have the 2:1 frame aspect or duplicated-scene geometry of a true stereo capture. Moondream appears to be keyword-matching "two halves" against any image with two roughly-symmetric foreground regions (TV-grid panels, paired faucets, paired ovals). Filed as Issue #1 in `docs/02_filtering_and_labeling.md`.

> [!WARNING]
> **Social-presence pass rate dropped to 0 / 100 (vs 3 / 100 on May 17).** The stereo gate's false positives killed every video that previously cleared YOLO. After the stereo gate is patched (Issue #1), the projected base yield is still ≤ 1 / 100 on the unscripted Ego4D first-100 slice — even the three videos that reached the multi-person Moondream gate were rejected (`0/2 confirmed across 5 attempts`), and a spot-check of the convenience-store-cashier video (`closest_pass_vlm1_2.jpg`) suggests one of those rejections is itself a false negative. Filed as Issue #2 with metadata-positive allowlist, stronger-VLM, and dataset-mix remediation options.

## Pipeline Configuration

| Setting | Value | Notes |
|---|---|---|
| Architecture | single-pass interleaved | Auto-selected (host RAM ≥ 48 GB) |
| `social_presence_pose` | `yolov8n-pose.pt` | Resolved via tier-per-host registry |
| `social_presence_hand_landmarker` | `models/mediapipe/hand_landmarker.task` | New per Resolved Issue #9 (MediaPipe Tasks API float16 bundle, ~7 MB) |
| `social_presence_vlm_verify` | `moondream` (`moondream:latest`, 1.7 GB Ollama) | Active; runs stereo + chin + multi-person gates |
| `SAF_VLM_VERIFY_SOCIAL` | `1` (default) | On per Resolved Issue #5 default |
| `sample_rate_fps` | `1/3` (1 frame per 3 s) | New default per Resolved Issue #11 |
| `filtering_vlm` | `qwen2.5vl:latest` (lazy fallback from `qwen2.5vl:7b`) | Per Resolved Issue #7; not actually invoked here (climax deferred to Layer 03 per Resolved Issue #8) |
| `TMPDIR` | `/Volumes/Extreme SSD/tmp` | Per Resolved Issue #2 of `01_dataset_acquisition.md` |
| `HF_HOME` | `/Volumes/Extreme SSD/huggingface_cache` | Per Resolved Issue #3 of `01_dataset_acquisition.md` |
| `return_hands` | **`True`** | First run since Resolved Issue #9 to exercise the production setting; no AttributeError. |

## Quantitative Results

### Registry build (Node 01)
- `scan_datasets()` indexed **8 107** total videos (Charades-Ego 7 860 / Ego4D 247 / EPIC-KITCHENS 0 / EgoProceL 0) — unchanged from May 17.
- The 100-video test slice was deterministic-sorted Ego4D-only (`ego4d[:100]`) and persisted to `test_video_registry.json` (**67.4 GB** combined; median file size **374 MB**; max **5.1 GB**). All 100 paths verified existing and non-zero before the filter run.

### Filtering (Node 02 social-presence stage)
| Metric | Value |
|---|---|
| Videos processed end-to-end | **100 / 100** |
| Errors during processing | **0** (validates Resolved Issue #9 — MediaPipe Tasks API is stable) |
| Skipped by metadata-solo prefilter (Resolved Issue #12) | **18 / 100** |
| YOLO+VLM full-pass | **82 / 100** |
| Rejected by stereo VLM gate (Resolved Issue #10) | **66 / 82** ← *see Issue #1, dominated by false positives* |
| Rejected by multi-person VLM gate (Resolved Issue #5) | **3 / 82** (each `0/2 confirmed across 5 attempts`) |
| No YOLO-pose head+shoulder positive at all | **13 / 82** |
| Passed social-presence filter | **0 / 100** (0.0 %) |
| Mean per-video YOLO+VLM time | **13.9 s** |
| Median per-video YOLO+VLM time | **5.7 s** |
| Max per-video YOLO+VLM time | **118.0 s** |
| Total wall time | **19.0 min** (9.5× faster than May 17) |

The May 17 run on the same first-100 slice yielded 3 / 100 (and 2 of those 3 were false positives — the stereo-rig craft video and the wearer-chin hedge-trimming GoPro). The metadata-solo prefilter intercepted those *type* of cases this run (the `0219271c` craft video was correctly skipped at intake on `scenarios=["Crafting"]`), but the introduction of the stereo VLM gate as a once-per-video Moondream call on the first pose-positive frame has over-fired catastrophically. End-effect is that the post-#12 filter pipeline finds zero positives in the first 100 Ego4D videos, including videos where YOLO-pose is genuinely firing on real bystanders (see the `closest_pass_vlm1` spot-check below).

## Discovered Bugs

Two new issues filed under the Bug Documentation Style Guide in `docs/02_filtering_and_labeling.md` → `⚠️ Unresolved Issues & Suggestions`:

1. **Issue #1 — Stereo VLM gate over-fires on ordinary single-camera egocentric video** (critical filter-quality regression — 66 of 82 processed videos rejected with the `Stereo gate rejected` log line; 4 spot-checked, all single-camera; recommended remediation is a cheap aspect-ratio prefilter `img_w / img_h >= 1.9` before invoking Moondream).
2. **Issue #2 — 0 / 100 social-presence pass rate on the May 18 Ego4D first-100 slice** (yield collapse — once Issue #1 is patched, projected yield is still ≤ 1 / 100 on this slice; recommended remediation is a metadata-social positive allowlist complementing Resolved Issue #12's solo allowlist).

Each issue records technical root cause, verification details, and three remediation options with Pros / Cons per the style guide. Awaiting selection.

## Qualitative Spot Check

### Closest-to-passing video (highest YOLO-pose+VLM signal, still rejected)
With 0 / 100 passing, the closest-to-passing case is the three Moondream-multi-person-rejected videos (all hit `0/2 confirmed across 5 attempts`). The most informative of the three is:
- **Video ID**: `0049fdd8-0044-4ef5-9c34-b3469416ebe5` (Ego4D, `scenarios=["Grocery shopping indoors"]`, ~22 min duration, 1920×1080)
- **Rejection log**: `[SocialPresenceDetector] VLM gate rejected 0049fdd8-…mp4: 0/2 confirmed across 5 attempts`
- **YOLO+VLM time**: 68.7 s (high because the stereo gate ran first and replied NO, then the multi-person gate burned 5 Moondream calls before dropping the video)
- **Spot Check Result**: ⚠️ **Likely false negative.** Sampled at 60 s / 180 s / 300 s (`closest_pass_vlm1_{1,2,3}.jpg`): the wearer is walking through a convenience-store aisle past a cashier in a red shirt visible *behind* the snack rack. YOLO-pose is correctly firing on the cashier's head+torso through the rack gaps, but Moondream rejects on the multi-person prompt because the cashier is partially occluded by Planters / Potato Skins / Hannaford bags in every sampled frame. A human spot-check accepts this as a real social context (wearer + cashier).

![Closest-pass frame 1 — convenience-store aisle, cashier in red behind snack rack](frames/closest_pass_vlm1_1.jpg)
![Closest-pass frame 2 — closer view of the snack rack with cashier partially visible](frames/closest_pass_vlm1_2.jpg)
![Closest-pass frame 3 — same scene, cashier still occluded by merchandise](frames/closest_pass_vlm1_3.jpg)

The other two Moondream-multi-person-rejected videos (`02c40de9-…` deserted register counter, `002c3b5c-ed8…` wearer alone in a tiled kitchen) are correctly rejected — see `closest_pass_vlm2_2.jpg` and `closest_pass_vlm3_2.jpg`.

### Worst-failing video (highest-confidence rejection that is actually wrong)
The clearest-cut wrong rejection is the stereo gate firing on a 4:3 single-camera GoPro:
- **Video ID**: `0030b1e9-c6a6-4809-a495-8d45791f9775` (Ego4D, `scenarios=["jobs related to construction/renovation company"]`, 164.8 s duration, **2560×1920**)
- **Rejection log**: `[SocialPresenceDetector] Stereo gate rejected 0030b1e9-…mp4: side-by-side stereoscopic capture detected`
- **YOLO+VLM time**: very short (stereo gate short-circuits on the first pose-positive frame)
- **Spot Check Result**: ❌ **False positive on stereo gate** (and, separately, would have been a true negative on social presence — the May 17 spot-check already established this is the wearer alone trimming a hedge, with the wearer's own chin peeking into the bottom edge of the frame). The frame is 4:3 aspect (1.33), not a stereo 2:1. The wearer's left and right arms appear roughly symmetrically positioned because they are gripping the two handles of a pair of hedge shears — Moondream is apparently mistaking that arm symmetry for "two halves of a stereo image rendered twice".

![Worst-fail frame 1 — single-camera 4:3 GoPro, hedge shears, wearer's chin at bottom](frames/stereo_fp_hedge_1.jpg)
![Worst-fail frame 2 — same scene, no second person](frames/stereo_fp_hedge_2.jpg)
![Worst-fail frame 3 — same scene mid-trimming](frames/stereo_fp_hedge_3.jpg)

The "wrong stereo gate" failure pattern repeats across many distinct scenes — a different sample:
- `002ad105-bd9a-…` (hotel room with YouTube on a wall TV, single-camera 1440×1080) — `stereo_fp_hotel_2.jpg`
- `000786a7-3f9d-…` (plumbing workshop, single-camera 2560×1920) — `extra_stereo_000786a7.jpg`
- `001e3e4e-…` (close-up of two ovals, single-camera 1920×1440) — `extra_stereo_001e3e4e.jpg`

![Stereo FP — hotel TV, single camera](frames/stereo_fp_hotel_2.jpg)
![Stereo FP — plumbing workshop, single camera](frames/extra_stereo_000786a7.jpg)
![Stereo FP — close-up of two ovals, single camera](frames/extra_stereo_001e3e4e.jpg)

### Sanity Check on Skips and True-Negative Rejections
- **Metadata-solo prefilter (Resolved Issue #12)**: 18 / 100 videos were skipped before YOLO. The known true-stereo-but-actually-solo `0219271c-…` paper-craft video from the May 17 report (`scenarios=["Crafting"]`) is correctly intercepted here — i.e., the metadata gate is doing exactly what it was specified to do. No YOLO+VLM budget was burned on it.
- **Multi-person VLM gate true negatives**: `02c40de9-…` (`closest_pass_vlm2_2.jpg`: deserted convenience-store register, wearer's phone visible at bottom) and `002c3b5c-ed8…` (`closest_pass_vlm3_2.jpg`: wearer alone in a tiled kitchen) are correctly rejected.
- **Resolved Issue #9 stability**: 0 errors across 82 full-pass videos with `return_hands=True` — the MediaPipe Tasks API HandLandmarker is stable end-to-end on `mediapipe==0.10.35`.

## Artifacts
- `e2e_social_filter_100.json` — per-video result records (skip reason, elapsed time, bystander aggregates, pass/fail)
- `test_video_registry.json` — input slice
- `frames/` — spot-check stills for closest-to-passing and worst-failing videos
- Source runner: `scratch/run_e2e_social_only_v3.py`
- Log: `scratch/e2e_v3.log`
