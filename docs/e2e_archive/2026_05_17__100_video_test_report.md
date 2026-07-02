# End-to-End Test Report: 100-Video Batch
**Date**: May 17, 2026
**Target Host**: Mac Studio (M4 Max, 64 GB Unified Memory)
**Nodes Tested**: 01 (Dataset Acquisition) & 02 (Filtering & Labeling)
**Tier (auto-detected)**: `medium`
**Registry composition**: 100 Ego4D `full_scale` videos (first 100 by `id` sort) — Charades-Ego excluded at intake per `FilteringPipeline._SUPPORTED_DATASETS = ("ego4d",)` (Resolved Issue #18).

## Executive Summary

The May 16 run's 4 unresolved issues have all been resolved in `9408488` and the new defaults were exercised end-to-end: VLM verify is now ON by default (`SAF_VLM_VERIFY_SOCIAL=1`), the Moondream prompt is tightened to exclude on-TV / on-screen / on-photograph faces (closing the May 16 hotel-TV false positive), `FilteringPipeline` now refuses non-Ego4D datasets at intake, Ollama tags resolve lazily against `ollama list`, and Layer 02 no longer runs the climax-extraction seek storm. **All four prior false-positive / silent-drop classes are gone.**

The new run surfaced **four new unresolved issues**, two of which are critical filter-quality regressions that I confirmed via spot-check:

> [!WARNING]
> **Both of the spot-checked passing videos are false positives.** The highest-scoring video (`0219271c`, score 843) is a side-by-side stereoscopic egocentric craft video where YOLO-pose double-counts the lone wearer across stereo halves. The lowest-scoring passing video (`0030b1e9`, score 14) is a single person trimming a hedge alone — YOLO is firing on the wearer's own chin/face peeking into the bottom edge of the GoPro frame. Filed as Issues #2 and #3 in `docs/02_filtering_and_labeling.md`.

> [!CAUTION]
> **`SocialPresenceDetector.detect(..., return_hands=True)` AttributeErrors on every video** under the installed `mediapipe==0.10.35`, because `mp.solutions` no longer exists in that release. `FilteringPipeline.process_video_vlm_pass` (`src/filtering_and_labeling/pipeline.py:285`) hard-wires `return_hands=True`, so the production Layer 02 entry point crashes on every Ego4D video before any bystanders can be persisted. The numbers below were obtained from a manually patched runner with `return_hands=False`. Filed as Issue #1.

## Pipeline Configuration

| Setting | Value | Notes |
|---|---|---|
| Architecture | single-pass interleaved | Auto-selected (host RAM ≥ 48 GB) |
| `social_presence_pose` | `yolov8n-pose.pt` | Resolved via tier-per-host registry |
| `social_presence_vlm_verify` | `moondream` (`moondream:latest`, 1.7 GB Ollama) | **Active** for this run |
| `SAF_VLM_VERIFY_SOCIAL` | `1` (new default after Resolved Issue #1) | Enabled by default; was overridden to `0` on May 16 |
| `TMPDIR` | `/Volumes/Extreme SSD/tmp` | Per Resolved Issue #2 of `01_dataset_acquisition.md` |
| `HF_HOME` | `/Volumes/Extreme SSD/huggingface_cache` | Per Resolved Issue #3 of `01_dataset_acquisition.md` |
| `return_hands` | **`False` (forced)** | Workaround for Issue #1 below; production callers use `True` and crash |

## Quantitative Results

### Registry build (Node 01)
- `scan_datasets()` indexed **8 107** total videos (Charades-Ego 7 860 / Ego4D 247 / EPIC-KITCHENS 0 / EgoProceL 0) — unchanged from May 16.
- The 100-video test slice was deterministic-sorted Ego4D-only (`ego4d[:100]`) and persisted to `test_video_registry.json` (**72.3 GB** combined; median file size **420 MB**; max **5.4 GB**). All 100 paths verified existing and non-zero before the filter run.

### Filtering (Node 02 social-presence stage, VLM verify ON)
| Metric | Value |
|---|---|
| Videos processed | **100 / 100** |
| Errors during processing | **0** (with `return_hands=False` workaround — see Issue #1) |
| Passed social-presence filter | **3 / 100** (3.0 %) |
| Confirmed true positives (spot-check) | **0 / 2 spot-checked** — see Issues #2 and #3 |
| Mean per-video YOLO+VLM time | **108.5 s** |
| Median per-video YOLO+VLM time | **75.8 s** |
| Max per-video YOLO+VLM time | **794.9 s** (`011ee98a`, 9 314 s duration) |
| Total wall time | **180.8 min** (3.0 h) |

The May 16 run on the mixed 80-Charades + 20-Ego4D registry reported 62 / 86 passing (72 %) with VLM verify *off*. The new 3 % rate reflects both (a) the dataset switch to Ego4D-only (`full_scale` Ego4D is dominated by unscripted solo activity) and (b) the tightened Moondream prompt doing its job. Independently spot-checked the longest-running rejected video (`011ee98a-…`, 9 314 s) and the wearer is alone the entire time doing paperwork — the gate is *not* over-rejecting in the obvious cases. The yield problem is filed as Issue #4 with a metadata-prefilter remediation path.

## Discovered Bugs

All four new issues are filed under the Bug Documentation Style Guide in `docs/02_filtering_and_labeling.md` → `⚠️ Unresolved Issues & Suggestions`:

1. **Issue #1 — `mediapipe.solutions` AttributeError crashes Node 02 on every video** (critical — production `FilteringPipeline` is currently broken because `mediapipe==0.10.35` removed the legacy namespace; `social_presence.py:66` and `pipeline.py:285` need to migrate to the Tasks API or version-pin).
2. **Issue #2 — Side-by-side stereo egocentric videos pass as false positives** (highest-scoring video in this run is a stereo capture of one person; YOLO double-counts the wearer across stereo halves; remediation: aspect-ratio detector at intake, crop to left half).
3. **Issue #3 — Wearer's own chin/face at frame bottom passes YOLO-pose and the VLM gate** (lowest-scoring passing video; the existing Anti-Wearer Heuristic doesn't catch "head-only at bottom edge" geometry; remediation: extend heuristic with shoulder-keypoint + bbox-height conditions).
4. **Issue #4 — 3 % pass rate on the first-100 Ego4D slice is operationally unworkable** (108 s/video × 33 negative videos per positive ≈ 10 host-days per 10 k corpus; remediation: prefilter on Ego4D `scenarios` person-count tags before paying the YOLO+VLM cost).

Each issue records technical root cause, verification details, and 2–3 remediation options with Pros / Cons per the style guide. Awaiting selection.

## Qualitative Spot Check

### Highest-Scoring Video (passed filter with strongest signal)
- **Video ID**: `0219271c-7641-4e17-a00c-81c42e0d4779` (Ego4D, paper-craft activity, ~17 min duration)
- **Social-presence score**: `843.32`
- **Bystander tracks**: 7, **frame detections total**: 984 (across the 17-min duration)
- **Top detection timestamps**: `955 s` (conf 0.96), `664 s` (conf 0.96), `383 s` (conf 0.95)
- **Spot Check Result**: ❌ **False positive.** All three sampled frames show the same scene rendered as a **side-by-side stereoscopic pair** (2880×1080 frame, ~2:1 aspect). The wearer is alone at a craft table cutting paper. YOLO-pose detects the wearer's own black-shirted torso + arms in each stereo half and treats them as two distinct people; the Moondream gate then confirms ("two people visible") because the stereo doubles are visually distinct and are not on a screen/photo. This single video accounts for **>93%** of the entire run's positive-detection mass (984 / ~1 062 frame-detections across all 3 passing videos).

![Best video frame 1 — stereo egocentric, wearer alone with craft](frames/best_frame_1.jpg)
![Best video frame 2 — stereo egocentric, wearer's hands and torso doubled](frames/best_frame_2.jpg)
![Best video frame 3 — stereo egocentric, wearer alone cutting paper](frames/best_frame_3.jpg)

This is a **new false-positive class** not covered by Resolved Issue #1's tightened Moondream prompt (which targets on-screen / on-TV / on-photo faces). Filed as Issue #2 with an aspect-ratio detector remediation.

### Lowest-Scoring Video That Still Passed (barely cleared the gate)
- **Video ID**: `0030b1e9-c6a6-4809-a495-8d45791f9775` (Ego4D, outdoor hedge trimming, GoPro head-mount)
- **Social-presence score**: `14.13`
- **Bystander tracks**: 3, **frame detections total**: 18 (clustered between `33 s` and `153 s`)
- **Spot Check Result**: ❌ **False positive.** All three sampled frames (`49 s`, `93 s`, `153 s`) show one person trimming a hedge alone outdoors. YOLO is detecting the **wearer's own chin / lower jaw** peeking into the bottom edge of the GoPro frame as the camera looks down at the shears. The Geometric Anti-Wearer Heuristic doesn't catch it because (a) the chin patch contains head keypoints (mouth, jaw) so it isn't an "edge bbox without a head," and (b) the bbox doesn't extend to the literal bottom pixel row. Moondream then confirms because there *is* technically a human face visible — it cannot distinguish wearer-chin from bystander-face from a small cropped patch.

![Worst-passer frame 1 — wearer's chin visible at bottom, hedge shears in hands](frames/worst_frame_1.jpg)
![Worst-passer frame 2 — same scene, no other person present](frames/worst_frame_2.jpg)
![Worst-passer frame 3 — same scene, only the wearer's chin and arms visible](frames/worst_frame_3.jpg)

Filed as Issue #3 with an extended Anti-Wearer Heuristic (head-only-at-bottom-edge rejection rule) as the recommended remediation.

### Sanity Check on Failed Videos
To verify the 3 % pass rate isn't pathologically over-rejecting, I spot-checked the **longest-running failed video** — the case where the VLM gate had the most chances to confirm and still rejected:
- **Video ID**: `011ee98a-afc2-4088-b4d1-e0f7d84f3611` (9 314 s = 2.6 h duration, 795 s filter time)
- **Sampled at 25 %, 50 %, 75 % of duration** (`2 328 s`, `4 657 s`, `6 985 s`)
- **Result**: ✅ Correctly rejected — all three frames show one person alone doing paperwork (writing on a printed form, with a newspaper underneath). The newspaper contains a small printed face photo, which the tightened Moondream prompt correctly ignores per Resolved Issue #1. The two visible arms are both the wearer's.

![Failed frame 1 — wearer doing paperwork alone, newspaper visible](frames/failed_frame_1.jpg)
![Failed frame 2 — wearer writing on form, no second person](frames/failed_frame_2.jpg)

So the gate is rejecting solo content correctly; the yield problem (Issue #4) is a real artefact of Ego4D's solo-dominant composition, not a filter-quality regression.

## Honest Caveats

1. **Production entry point is broken; the numbers above come from a patched runner.** With `return_hands=True` (the production setting), all 100 videos AttributeError-crash on `mp.solutions.hands` before any bystanders can be persisted. I obtained the numbers in this report by forcing `return_hands=False` in `scratch/run_e2e_social_only_v2.py`. The filter logic itself is unchanged — only the MediaPipe hand-detection branch is bypassed — so the pass/fail decisions and scores are representative of what production would emit *if* Issue #1 is fixed. But until then, **Node 02 produces zero usable manifest output**.
2. **Two of the three passing videos are false positives** confirmed by spot-check. The third passing video (`00eab18b`, score 50.34) was not spot-checked individually; its score is consistent with a real multi-person scene rather than a stereo or self-detection artefact, but this is unconfirmed.
3. **`SocialPresenceDetector` was invoked directly, not via `FilteringPipeline`**, because of Issue #1. Coverage of the full `process_video_vlm_pass` path (which includes metadata-driven task labeling for Ego4D) is therefore untested in this run. That said, the social-presence gate is what determines whether a video reaches the labeling stage at all, and the gate is what's broken.
4. **No videos were dropped due to file-system errors** — all 100 Ego4D paths existed and read successfully. This is a regression from the May 16 run, where 14 large Ego4D videos couldn't complete in the session window; the Resolved Issue #4 fix (removing the climax-extraction seek storm from Layer 02) is doing its job here.

## Raw Artifacts
- Per-video filter output: `e2e_social_filter_100.json` (100 entries, ~75 KB)
- Test registry: `test_video_registry.json` (100 Ego4D entries)
- Spot-check frames: `frames/best_frame_{1,2,3}.jpg`, `frames/worst_frame_{1,2,3}.jpg`, `frames/failed_frame_{1,2}.jpg`
- Helper script: `scratch/run_e2e_social_only_v2.py` (forced `return_hands=False` workaround)
- Bug documentation updates: `docs/02_filtering_and_labeling.md` Issues #1 – #4
