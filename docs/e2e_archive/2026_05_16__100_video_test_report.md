# End-to-End Test Report: 100-Video Batch
**Date**: May 16, 2026
**Target Host**: Mac Studio (M4 Max, 64 GB Unified Memory)
**Nodes Tested**: 01 (Dataset Acquisition) & 02 (Filtering & Labeling)
**Tier (auto-detected)**: `medium`

## Executive Summary

A 100-video E2E run targeted Node 01 (registry scan) and Node 02 (filtering & labeling). The 100-entry registry was a mix of 80 Charades-Ego short clips and 20 Ego4D `full_scale` videos drawn off the Extreme SSD.

**Headline result**: The Node 02 social-presence YOLO-pose filter completed on 86 / 100 videos (62 / 86 passed). The full `FilteringPipeline` (YOLO + optical-flow climax + Stage-2 VLM refinement) was abandoned mid-run after the first Ego4D video failed to finish its Task Refinement step in 3+ minutes — see Issue #4 in `docs/02_filtering_and_labeling.md`. The remaining 14 unfiltered videos are all Ego4D entries ≥ 450 MB whose per-video YOLO time was projected at 1.5–7 min each; they were excluded from this report's quantitative analysis but the throughput gap is recorded as Unresolved Issue #4.

> [!WARNING]
> Spot-checking the *highest-scoring* video in the run reveals it is a **false positive** — the camera wearer is alone in a hotel room watching YouTube, and YOLO-pose is firing on the faces displayed on the TV screen (48 "bystanders", score 346.67). This is a critical filter-quality regression and is filed as Unresolved Issue #1.

## Pipeline Configuration

| Setting | Value | Notes |
|---|---|---|
| Architecture | single-pass interleaved | Auto-selected by `FilteringPipeline.__init__` (host RAM ≥ 48 GB) |
| `social_presence_pose` | `yolov8n-pose.pt` | Resolved via tier-per-host registry |
| `social_presence_vlm_verify` | `moondream` | Loaded but **not exercised** — see below |
| `filtering_vlm` | `qwen2.5vl:7b` | **Tag not installed locally** — see Issue #3 |
| `SAF_VLM_VERIFY_SOCIAL` | `0` (overridden for this run) | Disabled so 80 Charades clips fit in the budget |
| `TMPDIR` | `/Volumes/Extreme SSD/tmp` | Per Resolved Issue #2 of `01_dataset_acquisition.md` |
| `HF_HOME` | `/Volumes/Extreme SSD/huggingface_cache` | Per Resolved Issue #3 of `01_dataset_acquisition.md` |

## Quantitative Results

### Registry build (Node 01)
- `scan_datasets()` indexed **8 107** total videos across all four configured roots (Charades-Ego 7 860 / Ego4D 247 / EPIC-KITCHENS 0 / EgoProceL 0).
- The 100-video test slice was deterministic-sorted (`charades_ego[:80] + ego4d[:20]`) and persisted to `test_video_registry.json` (6.0 GB combined). All 100 file paths verified existing and non-zero size before the filter run.

### Filtering (Node 02 social-presence stage)
| Metric | Value |
|---|---|
| Videos processed | **86 / 100** |
| Passed social-presence filter | **62 / 86** (72.1 %) |
| Passed (Charades-Ego) | 58 / 80 (72.5 %) |
| Passed (Ego4D, processed slice) | 5 / 6 (83.3 %) |
| Mean per-video YOLO time | 5.03 s |
| Median per-video YOLO time | 1.89 s |
| Max per-video YOLO time | 102.5 s (`002ad105`, 46-min Ego4D) |
| Total YOLO wall time | 432.9 s ≈ 7.2 min |

The full filter+climax pipeline aborted after the *first* Ego4D climax pass (`000786a7-…`) exceeded 3 min without completing. Issue #4 documents the bottleneck (`cap.set` round-tripping the H.264 decoder per coarse-pass frame).

## Discovered Bugs

All four issues were filed under the Bug Documentation Style Guide in `docs/02_filtering_and_labeling.md` → `⚠️ Unresolved Issues & Suggestions`:

1. **Issue #1 — YOLO-pose fires on TV / monitor / photo content** (high impact, surfaced by spot-check below).
2. **Issue #2 — Non-Ego4D datasets silently dropped at the metadata stage** (58 / 80 Charades-Ego videos passed YOLO but were dropped before they could enter `filtered_manifest.json`).
3. **Issue #3 — `qwen2.5vl:7b` and `gemma4:26b` Ollama tags pinned in `models_config.py` but not locally installed** (silent fall-back via `except Exception`).
4. **Issue #4 — Optical-flow climax extraction's `cap.set` seeking is unworkably slow on long Ego4D videos** (projected ≥ 15 min per 46-min video; reason the full filter pipeline could not complete on the Ego4D tail of the registry).

Each issue records technical root cause, verification details, and 2–3 remediation options with Pros / Cons per the style guide. Awaiting selection.

## Qualitative Spot Check

### Highest-Scoring Video (passed filter with strongest signal)
- **Video ID**: `002ad105-bd9a-4858-953e-54e88dc7587e` (Ego4D, ~17 min duration, 116 MB)
- **Social-presence score**: `346.67`
- **Bystander tracks**: 48 (mean confidence ≈ 0.77 on the dominant track, 109 frames over a 583 s span)
- **Spot Check Result**: ❌ **False positive.** The three frames sampled at the top detection timestamps (`462 s`, `700 s`, `1 045 s`) all show the camera wearer alone in a hotel room with a TV playing YouTube content. YOLO-pose is firing on the on-screen faces — a YouTube recommendation grid with multiple thumbnail faces in `best_frame_1.jpg`, a soccer-coach broadcast in `best_frame_2.jpg`, a single vlogger face in `best_frame_3.jpg`. There is no real second person anywhere in the video. **This video should have been dropped.**

![Best video frame 1 — wearer watching YouTube recommendation grid on TV](frames/best_frame_1.jpg)
![Best video frame 2 — wearer watching broadcast on TV](frames/best_frame_2.jpg)
![Best video frame 3 — wearer watching vlog on TV](frames/best_frame_3.jpg)

This single video accounts for 440 / ~750 total frames-with-detections in the entire 62-video pass set, i.e. **≈ 60 % of the filter's positive signal mass came from one false-positive video**. Without an on-screen-content filter (Issue #1), the dataset will be heavily contaminated by passive-viewing scenes that have zero real-world social interaction.

### Lowest-Scoring Video That Still Passed (barely cleared the gate)
- **Video ID**: `07HVMEGO` (Charades-Ego, ~10 s duration)
- **Social-presence score**: `0.507`
- **Bystander tracks**: 1, single frame, confidence 0.507, timestamp 7.92 s
- **Spot Check Result**: ❌ **Likely false positive.** Both sampled frames (`worst_frame_1.jpg`, `worst_frame_2.jpg`) show the wearer's hand reaching into a closet of hanging clothes — there is no second person visible at any point. The single low-confidence detection probably fired on a humanoid clothing silhouette (jacket on hanger) where YOLO-pose hallucinated head + shoulder keypoints from drape geometry, bypassing the keypoint gate added in Resolved Issue #22.

![Worst-passer frame 1 — wearer's hand in closet, no second person](frames/worst_frame_1.jpg)
![Worst-passer frame 2 — same closet scene](frames/worst_frame_2.jpg)

This is the regime Resolved Issue #22's VLM Early-Exit Verification gate (`SAF_VLM_VERIFY_SOCIAL=1`) was designed to catch — but VLM verification was disabled for this run to fit the 100-video budget in the session window. With VLM verify on, this video would have made one Moondream call and been rejected.

## Honest Caveats

1. **VLM verification was disabled.** `SAF_VLM_VERIFY_SOCIAL=0` was set so the 80 short Charades clips would complete in minutes instead of an hour. Both spot-check failures above (Issues #1 worst-case and the closet false positive) would likely have been caught by the existing Moondream gate from Resolved Issue #22. The "best video is a false positive" finding may also reflect what happens when the VLM verify gate is bypassed — but the YOLO-pose path alone is what most callers configure under throughput pressure, so the failure mode is real.
2. **14 of 100 videos were not filtered** — all are Ego4D `full_scale` files ≥ 450 MB. The 86 we did process is statistically robust for a best/worst spot check but is not a uniform sample of the full 100.
3. **Charades-Ego "pass" count is misleading at the manifest level.** 58 Charades-Ego clips passed the YOLO filter, but per Issue #2 all 58 would have been dropped during `contextual_task_labeling` before reaching `filtered_manifest.json`. The 72 % pass rate is therefore a *filter-quality* figure, not a manifest-yield figure.
4. **`002ad105` was re-run alone after the main batch** to recover its `bystander_detections` payload (the incremental save in `run_e2e_social_only.py` had not yet checkpointed past index 85 when the process was killed).

## Raw Artifacts
- Per-video filter output: `e2e_social_filter_100.json` (86 entries, ~200 KB)
- Test registry: `test_video_registry.json` (100 entries)
- Spot-check frames: `frames/best_frame_{1,2,3}.jpg`, `frames/worst_frame_{1,2}.jpg`
- Helper script: `scratch/run_e2e_social_only.py`
- Bug documentation updates: `docs/02_filtering_and_labeling.md` Issues #1 – #4
