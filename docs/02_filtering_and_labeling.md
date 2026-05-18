# AI Task Breakdown: Video Filtering and Labeling

## Objective
To ensure computational efficiency and relevance, not all downloaded videos make it to the advanced Social Feature Layers. This module performs two critical filtration passes to weed out unusable data from first-person (POV) perspectives.

---

## 1. The Social Presence Filter
Since our project relies on investigating social-affective interactions, a video of a person entirely alone is useless. 

**Criteria**: 
The video **must** contain more than one person. Since this is an egocentric/POV video, the person whose perspective is being shown (the camera wearer) *does not count* towards this total. There must be at least one other visible human actor present in the frame. 

To achieve this, the system uses a **Multi-Modal Verification Pipeline**:
- **Pose Keypoint Validation (YOLOv8-Pose)**: The system utilizes `yolov8n-pose.pt`. Detections are only kept if they contain at least one head keypoint (nose, eye, or ear) AND at least one shoulder keypoint above a minimum confidence threshold. This effectively eliminates false positives from the wearer's own outstretched limbs or vaguely humanoid equipment.
- **VLM Early-Exit Verification Gate**: Pose-confirmed candidate frames are sent to a fast Vision-Language Model (e.g., `moondream`) to answer a simple multi-person YES/NO prompt. The pipeline tracks consecutive VLM confirmations. Once a video hits the required consistency threshold, the filter short-circuits (early-exit) and marks the video as kept.
- **Geometric Anti-Wearer Heuristic**: As a baseline defense, the system drops detections touching the bottom edge of the frame without a head/torso, or full-height detections starting at the absolute top (`y1=0`).
- **Temporal Consistency**: A video is only "KEPT" if social presence is detected in at least **2 sampled frames** to filter out momentary glitches.

> [!IMPORTANT]
> **Integration Note**: To optimize storage on the "Extreme SSD", this filter is integrated directly into the **Dataset Acquisition** module via a **batched UID strategy**. Videos are downloaded in small groups (e.g., 50), filtered immediately, and non-social videos are purged. A persistent `processed_uids.json` tracks completion so that purged videos are never re-downloaded.

**Task Requirements**:
- Evaluate the raw videos from the initial manifest.
- Drop any video where zero exterior individuals are found.
- **Persist bounding boxes**: For every frame sampled during detection, the bounding box coordinates and timestamps of detected `person` instances **must be written to the output manifest**, not just the keep/drop decision. Downstream Social Layers (03a Attention, 03b Reasonable Emotion, etc.) depend on this data to isolate and crop bystander regions without re-running detection.
- *Potential Implementation Idea*: Utilize standard lightweight Object Detection (e.g. YOLOv8) on heavily downsampled frames to check for bounding boxes labeled `person`.

---

## 2. Contextual Task Labeling
A social interaction has fundamentally different mechanics if the POV person is reading a book versus cooking a meal with a knife. For downstream layers to accurately process features (like "Flinch" physics), we must define the task context. 

Because unstructured Ego4D videos can be lengthy, a single video may contain **multiple distinct tasks**.

**Criteria**:
We must actively generate task labels spanning the duration of the clip by extracting them from the **Ego4D Metadata**. If no clear tasks are identified in the metadata (e.g., descriptions like "Idling" or "No clear task") for the entire duration, the video is dropped.

**Task Requirements**:
- **Metadata Extraction**: Parse the Ego4D `scenarios` and `descriptions` (from `ego4d.json` and benchmark-specific annotation files) to classify the sequential actions.
- **Taxonomy Mapping**: Map the natural language descriptions and scenario tags to a structured array of `identified_tasks`.
- **Velocity Mapping**: Since metadata doesn't explicitly provide "velocity," we use a predefined lookup table to map task labels to physical velocities:
  - **Fast / Instinctual**: slipping, dropping, throwing, impact.
  - **Medium**: standard manual labor, walking, talking.
  - **Slow / Cognitive**: reading, writing, thinking, solving.
- *Drop Rule*: If the metadata lists "Idling," "No clear task," or "Ambiguous activity" (or is missing task-level annotations) for the entire video duration, it must be discarded.

---

## 3. Temporal Task Climax Identification
Knowing *what* the task is isn't enough; downstream affective layers need to know exactly *when* it happens so they can measure bystander reactions accurately.

**Criteria**:
Identify the exact timestamp or narrow temporal window where the "climax" or core action of each task occurs (e.g., the exact moment the dropped glass shatters, or the exact moment the basketball leaves the hands).

**Task Requirements**:
- **Hybrid Optical Flow + VLM Climax Detection**: A two-stage approach that requires zero additional model downloads:
  - **Stage 1 — Optical Flow Variance Peak**: Use OpenCV's `cv2.calcOpticalFlowFarneback` to compute dense optical flow. To optimize throughput and precision, this is implemented as a **Two-Pass Hierarchical Detection**: a coarse pass at ~5 FPS identifies the rough peak window within the task's temporal bounds, followed by a native 30 FPS dense pass within ±1s of the coarse peak for full temporal precision. The frame with the highest flow magnitude is the kinetic `task_climax_sec`. This is highly accurate for abrupt physical actions (dropping, slipping, throwing).
  - **Stage 2 — VLM Refinement (slow/cognitive tasks)**: For tasks classified as `"slow"` velocity (e.g., solving a puzzle, reading a sign), sample 3-5 candidate frames around the optical flow peak and use the existing **moondream** VLM to score each frame's proximity to the action climax, using the `task_label` as context.
- **Dynamic Action Velocity & Delay Buffers**: Human reactions vary wildly depending on the abruptness of a task. The system maps the metadata-derived `task_label` to a "Velocity" class. We use this to compute a dynamic `task_reaction_window_sec`:
  - **Fast / Instinctual** (e.g., slipping, dropping an item): Requires a short buffer. Climax + `[0.5s to 2.0s]`.
  - **Medium** (e.g., throwing a ball, standard interaction): Climax + `[1.0s to 3.0s]`.
  - **Slow / Cognitive** (e.g., solving a puzzle, reading a sign): Climax + `[2.0s to 6.0s]`.

> [!IMPORTANT]
> **Execution Location**: Climax extraction is no longer executed by Layer 02 itself. Per Resolved Issue #8 below, Layer 02 emits `task_temporal_metadata = {}`, and the first Layer 03 pipeline to consume the manifest invokes `shared.climax_extraction.populate_climax_for_manifest()` to fill in the optical-flow peak, dynamic reaction window, and optional VLM refinement in place. Subsequent Layer 03 pipelines find the metadata cached and skip the optical-flow pass entirely.

---

## Output
This module outputs a `filtered_manifest.json` that the downstream **Social Feature Extraction Layers** will consume. Only clips that survived both the Social Presence check and Contextual Task extraction make it onto this ledger.

### `filtered_manifest.json` Schema Definition
All downstream Social Feature Layers depend on this exact contract. Any additions to this schema must be **additive only**—never remove or rename existing keys.

```json
{
  "video_id": "ego4d_clip_10293",
  "source_dataset": "ego4d",
  "video_path": "/Volumes/Extreme SSD/raw_videos/ego4d_clip_10293.mp4",
  "fps": 30.0,
  "duration_sec": 45.0,
  "identified_tasks": [
    {
      "task_id": "t_01",
      "task_label": "Chopping vegetables",
      "task_confidence": 0.92,
      "task_velocity": "medium",
      "task_start_sec": 0.0,
      "task_end_sec": 45.0,
      "task_temporal_metadata": {
        "task_climax_sec": 5.2,
        "task_reaction_window_sec": [6.2, 8.2],
        "climax_extraction_method": "optical_flow_peak+vlm_refinement",
        "optical_flow_peak_magnitude": 42.8,
        "vlm_climax_confidence": 0.85
      }
    },
    {
      "task_id": "t_02",
      "task_label": "Dropping a plate",
      "task_confidence": 0.88,
      "task_velocity": "fast",
      "task_start_sec": 0.0,
      "task_end_sec": 45.0,
      "task_temporal_metadata": {
        "task_climax_sec": 24.1,
        "task_reaction_window_sec": [24.6, 26.1],
        "climax_extraction_method": "optical_flow_peak",
        "optical_flow_peak_magnitude": 78.3
      }
    }
  ],
  "bystander_detections": [
    {
      "person_id": 0,
      "timestamps_sec": [0.0, 0.5, 1.0, 1.5],
      "bounding_boxes": [
        [120, 80, 340, 510],
        [125, 78, 345, 512],
        [130, 82, 350, 515],
        [128, 80, 348, 513]
      ],
      "detection_confidence": [0.94, 0.91, 0.93, 0.90]
    }
  ],
  "hand_detections": [
    {
      "timestamp_sec": 0.0,
      "hand_boxes": [
        [410, 520, 560, 670],
        [610, 540, 760, 690]
      ]
    },
    {
      "timestamp_sec": 0.5,
      "hand_boxes": [
        [415, 522, 565, 672]
      ]
    }
  ]
}
```

> **Co-indexing rule**: Within each `bystander_detections` entry, `timestamps_sec[i]`, `bounding_boxes[i]`, and `detection_confidence[i]` are strictly co-indexed. The i-th element of each array describes the same sampled frame.

> **`hand_detections` contract**: One entry per sampled frame that contained MediaPipe-detected wearer hands. `timestamp_sec` matches a sampled frame timestamp from `bystander_detections`, and `hand_boxes` is a list of `[x1, y1, x2, y2]` integer pixel coordinates. Frames with no detected hands are simply omitted. Layer 03a (Attention) consumes this field for occlusion suppression — never remove or rename it.

## Verification & Validation Check
To ensure the filtering mechanisms are robust and correct:
- **Singular Video Test**: Execute the filter module against a single chosen `.mp4` video that is visually verified to have multiple interacting people. Inspect the generated JSON output to confirm the `bystander_detections` bounding boxes accurately surround the actors.
- **Batch Test**: Run the node over a subset folder (e.g., 50 videos). Aggregate the output to check distribution metrics (e.g., % dropped due to no social presence, % dropped due to idling). Assert that every entry in the `filtered_manifest.json` has `identified_tasks` with non-empty attributes, effectively proving the system scaled accurately without unhandled `None` crashes. Furthermore, utilize the **Mac Studio (M4 Max, 64 GB unified memory)** environment effectively by ensuring the batched VLM calls stay within the unified-memory budget. With 64 GB available, both YOLOv8 and Qwen2.5-VL co-reside, so the pipeline runs a single-pass interleaved architecture on hosts with ≥48 GB unified memory (see Resolved Issue #19), falling back to the two-pass architecture from Resolved Issue #11 on smaller hosts such as the 24 GB Mac mini.

---

## 🚀 Implementation Status

The filtering and labeling pipeline is fully operational within the `src/dataset_acquisition` and `src/filtering_and_labeling` modules.
- **Social Presence Filter**: YOLOv8-Pose (`yolov8n-pose.pt`) with head+shoulder keypoint validation, the geometric Anti-Wearer Heuristic, and a screen-aware Moondream verification gate (see Resolved Issue #22 and Resolved Issue #5).
- **Contextual Task Labeling**: Ego4D Metadata Extraction from `ego4d.json` is the only labeling path; non-Ego4D entries are skipped at intake until per-dataset labelers are written (Resolved Issue #6).
- **Temporal Climax Identification**: Implementation lives in `src/shared/climax_extraction.py` and is invoked by the first Layer 03 pipeline to consume the manifest, not by Layer 02 itself (Resolved Issue #8).

## 🧪 Resolved Issues & Implementation Refinements

1. **Pipeline Resumability & Error Isolation (Resolved - April 22)**:
   - **Problem**: Long batch runs were fragile; a single error in one video would crash the entire process.
   - **Solution**: Implemented incremental state saving (write-after-each-video) and isolated per-video errors to `02_filter_errors.json`.

2. **Unified Memory Constraints & Dual-Architecture Filtering (Resolved - May 02 / May 13)**:
   - **Problem**: The pipeline originally interleaved YOLO and VLM calls per video, which forced both models to reside in memory, risking OOM kills on 24GB unified memory hosts. However, enforcing a strict two-pass architecture (YOLO pass on all videos -> unload -> VLM pass) wasted SSD read bandwidth on 64GB hosts.
   - **Solution**: The pipeline now implements a memory-gated dual architecture. On hosts with ≥48 GB unified memory, it uses a **single-pass interleaved architecture** to process both YOLO and VLM steps in one read. On hosts with <48 GB, it automatically falls back to the **two-pass architecture**, safely isolating model memory footprints.

3. **VLM Model Tiering by Host Capacity (Resolved - May 13)**:
   - **Problem**: Stage-2 climax refinement was pinned to Qwen2.5-VL 3B to fit on 24GB hosts, leaving accuracy on the table for 64GB hosts.
   - **Solution**: Promoted the 7B tier to the default on 64GB hosts. The pipeline integrates with the central tier-per-host registry to auto-select `qwen2.5vl:7b` (≥48 GB) or `qwen2.5vl:3b` (<48 GB).

4. **MediaPipe Graph Initialization Crash on Apple Silicon (Resolved - May 14)**:
   - **Problem**: `MediaPipe` 0.10.11 failed due to `protobuf` incompatibilities on Apple Silicon.
   - **Solution**: Downgraded to `0.10.9` to restore stability for the wearer occlusion suppression layer.

5. **Screen-Aware VLM Verification Prompt (Resolved - May 17)**:
   - **Problem**: During the May 16 100-video E2E run, YOLO-pose's head+shoulder gate (`SocialPresenceDetector._has_head_and_shoulder()`, `src/shared/social_presence.py:101`) confidently fired on people rendered on TVs, monitors, and framed photographs because the on-screen faces genuinely satisfy the keypoint criterion. The geometric Anti-Wearer Heuristic only rejects bottom-edge / top-edge limb stubs, leaving no gate for "real 3D body vs pixel-rendered display surface". The single highest-scoring video in the run (Ego4D `002ad105-…`) was the camera wearer alone in a hotel room watching YouTube; all 48 "bystander" tracks were faces on the TV. The pre-existing Moondream YES/NO prompt in `_vlm_confirms_multiple_people` asked only "two or more distinct people" without excluding screens, so even with `SAF_VLM_VERIFY_SOCIAL=1` the VLM would have confirmed.
   - **Solution**: Tightened the Moondream prompt in `src/shared/social_presence.py` to explicitly require "two or more real, physical people actually present in the scene" and to enumerate the negative classes ("Do NOT count people shown on TVs, monitors, phone screens, photographs, posters, paintings, magazines, or reflections in mirrors"). The pipeline default (`SAF_VLM_VERIFY_SOCIAL=1` in `FilteringPipeline.__init__`) was already on — this resolution fixes the prompt content so that switching it back on actually filters out the on-TV class. Moondream remains the lightweight VLM (no new model weights); per-candidate-frame latency is unchanged from the original verification gate.

6. **Non-Ego4D Datasets Skipped at Intake Pending Per-Dataset Labelers (Resolved - May 17)**:
   - **Problem**: `FilteringPipeline.contextual_task_labeling` looked every video up in `self.metadata` (loaded from `EGO4D_METADATA_PATH`) and returned `[]` whenever the UID was not an Ego4D `video_uid`. In the May 16 E2E run, 58 of 80 Charades-Ego videos cleared the YOLO social filter and were then silently dropped from `filtered_manifest.json` because `process_video_vlm_pass` returns `None` on empty `identified_tasks`. The acquisition layer continued advertising four "Core Supported Datasets" (Ego4D, Charades-Ego, EPIC-KITCHENS, EgoProceL), masking the gap.
   - **Solution**: Added a `_SUPPORTED_DATASETS = ("ego4d",)` allowlist plus an `_is_supported_dataset()` gate at the top of `_run_single_pass` and `_run_two_pass` in `src/filtering_and_labeling/pipeline.py`. Non-Ego4D entries are now skipped before YOLO runs (avoiding the wasted social-filter pass) with an explicit per-video log line naming the unsupported dataset. The metadata-only ground-truth path (`task_confidence = 1.0`) is preserved for Ego4D; per-dataset labelers for Charades-Ego / EPIC-KITCHENS / EgoProceL are tracked separately for when their annotation parsers are written.

7. **Lazy Ollama Tag Resolution Against Local `ollama list` (Resolved - May 17)**:
   - **Problem**: The tier registry pinned the Stage-2 climax-refinement VLM to `qwen2.5vl:7b` on `medium`/`large` hosts (`src/models_config.py`), but the production Mac Studio had only `qwen2.5vl:latest` installed locally. Every slow-velocity VLM refinement call failed with `Error: model 'qwen2.5vl:7b' not found`, was swallowed by the broad `except Exception` in the climax stage, and silently fell back to `optical_flow_peak` without `vlm_climax_confidence`. The same latent bug existed for `layer_03b_ollama` (`gemma4:26b` registered, only `gemma4:latest` installed) and would fail identically once 03b ran end-to-end.
   - **Solution**: Added `_resolve_ollama_tag()` in `src/models_config.py`. `get_model()` now checks whether the configured tag (`qwen2.5vl:*`, `gemma4:*`, `moondream:*`) is present in `ollama list`; if the exact tag is missing but `<base>:latest` is installed, it substitutes `:latest` and emits a one-shot stderr warning so the fallback is visible without hiding genuine missing-model errors. The local-tag set is cached for the process lifetime so the `ollama.list()` call happens at most once per pipeline run. Hosts without the `ollama` Python client gracefully degrade to the original behavior (no fallback, identical to before).

8. **Climax Extraction Deferred to Layer 03 Co-Resident Pass (Resolved - May 17)**:
   - **Problem**: `temporal_climax_identification` issued a `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)` for every coarse-pass frame at ~5 FPS over each task range plus a dense pass at native FPS in the ±1 s window around the coarse peak. On Ego4D `full_scale` MP4 files served from the Extreme SSD, each seek triggered a fresh H.264 decoder reset because OpenCV does not retain GOP state across `cap.set` calls. In the May 16 E2E run the first 415 s Ego4D video had not finished its Task Refinement step after 3+ minutes; the 2 758 s videos in the registry projected to >15 minutes of climax extraction per video. With non-Ego4D datasets already skipped (Resolved Issue #6), this throughput limit alone capped the realistic dataset size at hundreds of videos per host-day.
   - **Solution**: Removed the optical-flow pass from `FilteringPipeline.process_video_vlm_pass` and extracted the two-pass coarse/dense flow + slow-task VLM refinement logic into `src/shared/climax_extraction.py`. Layer 02 now emits `identified_tasks` with `task_temporal_metadata = {}`; any Layer 03 pipeline that needs the reaction window calls `shared.climax_extraction.populate_climax_for_manifest(manifest_path, vlm_model=…)` (or `compute_task_climax_for_video()` when it already holds an open `cv2.VideoCapture`) before its own feature extraction. The shared utility is idempotent — tasks with non-empty metadata are skipped — so only the first Layer 03 to run for a given manifest pays the cost, and subsequent layers consume the cached metadata. The cross-layer schema contract change is acknowledged: `filtered_manifest.json` is a partial record until at least one Layer 03 has been invoked.


## ⚠️ Unresolved Issues & Suggestions

### Issue 1: `mediapipe.solutions` AttributeError crashes Node 02 on every video
**Status**: ⚠️ Confirmed Unresolved — Verified during the May 17 100-video Ego4D E2E run. `src/shared/social_presence.py:66` references `mp.solutions.hands` inside the lazy `mp_hands` property, but the installed `mediapipe==0.10.35` no longer exposes the `solutions` namespace at the top-level module (only `tasks`, `Image`, `ImageFormat`). The first VLM-positive frame for every video raises `AttributeError: module 'mediapipe' has no attribute 'solutions'`, which is caught by `SocialPresenceDetector.detect()`'s outer except and returns empty bystander lists — but `FilteringPipeline.process_video_vlm_pass` (`src/filtering_and_labeling/pipeline.py:285`) hard-wires `return_hands=True`, so the production Node 02 pipeline crashes on every single Ego4D video instead of falling back. Confirmed by running `scratch/run_e2e_social_only_v2.py` with `return_hands=True`: 100/100 videos errored before any pose detections could be saved. The standalone runner only succeeded after I forced `return_hands=False`.

**Option A (recommended)**: **Migrate `mp_hands` to the MediaPipe Tasks API** — Replace the legacy `mp.solutions.hands.Hands(...)` constructor with `mp.tasks.vision.HandLandmarker.create_from_options(...)` and update the per-frame call site (`social_presence.py:240`) to use the Tasks-style `detect()` signature (numpy → `mp.Image` wrapper, `.hand_landmarks` instead of `.multi_hand_landmarks`).
  - *Pros*: Forward-compatible with all `mediapipe` 0.10.x+ releases; removes the silent-crash hazard for the rest of the project lifetime; no version pinning needed in `ml_dependencies.md`.
  - *Cons*: Requires downloading the `hand_landmarker.task` model file (~7 MB) once and resolving its path through `models_config.py`; per-frame call signature and landmark coordinate shape both change, so `_extract_hand_boxes()` needs a small rewrite + re-test.

**Option B**: **Pin `mediapipe<0.11` and import-guard the lazy property** — Add `mediapipe<0.11` to `ml_dependencies.md` (the last release that exposes `mp.solutions`) and wrap `mp.solutions.hands` access in a `getattr(mp, 'solutions', None)` check that disables hand detection gracefully when the namespace is missing.
  - *Pros*: ~5-line patch; no need to download or vendor a new `.task` model.
  - *Cons*: `mediapipe.solutions` is already on the upstream deprecation path — pinning just delays the fix; silently disabling hand detection makes Layer 03e (affirmation gesture) silently degrade with no operator-visible signal.

**Option C**: **Make `return_hands=True` opt-in and default `FilteringPipeline` to `False`** — Change the `pipeline.py:285` call site to `return_hands=False` and require Layer 03e to call `SocialPresenceDetector` itself when it needs hands. Combine with import-guarding (Option B's guard) for safety.
  - *Pros*: Decouples the social-presence filter from MediaPipe entirely, restoring Node 02 throughput today; Layer 03e can be migrated to the Tasks API in isolation.
  - *Cons*: Duplicates YOLO-pose passes (Layer 02 + Layer 03e each run their own); doesn't actually fix the underlying API mismatch, only re-routes it.

Your selection: _____

---

### Issue 2: Side-by-side stereo egocentric videos pass the filter as false positives
**Status**: ⚠️ Confirmed Unresolved — The highest-scoring video in the May 17 Ego4D run, `0219271c-7641-4e17-a00c-81c42e0d4779` (score `843.32`, 984 frame-detections, 7 "bystanders"), is a side-by-side stereoscopic egocentric capture of a single person doing paper crafts alone at a table. Spot-checked frames at timestamps `955 s`, `664 s`, and `383 s` (`e2e_reports/2026_05_17/frames/best_frame_{1,2,3}.jpg`) all show the same scene rendered twice across the 2880×1080 frame — left half and right half are the stereo pair. YOLO-pose detects the wearer's own arms + black-shirted torso in each stereo half and treats them as **two separate persons**, and the tightened Moondream prompt from Resolved Issue #1 also confirms ("two people visible") because the duplicate visually-distinct humans are not on a screen or in a photograph — they are *literally rendered side-by-side in the frame*. This single video accounts for >93% of the entire run's positive-detection mass (984 / ~1062 total frame-detections across 3 passing videos). Without a stereo-format detector, any Ego4D entry shot with a stereo rig will pump false positives into `filtered_manifest.json`.

**Option A (recommended)**: **Detect 2:1 aspect-ratio stereo at intake and crop to the left half** — In `SocialPresenceDetector.detect()`, after `cap.read()`, check whether `width / height >= 2.0`; if so, crop to `frame[:, :width//2, :]` before passing to YOLO-pose. Cache the stereo flag on the manifest entry so downstream layers (03a Attention, 03d Proxemics) also see the cropped half.
  - *Pros*: One-line aspect-ratio test + slice catches the entire class deterministically; no extra model, no extra wall-time; preserves the manifest's per-frame bbox contract because all coordinates are already in the cropped half's coordinate system.
  - *Cons*: A small number of legitimately ultrawide (non-stereo) panoramic GoPro captures may exist in the wider Ego4D corpus; cropping them halves the FOV and may miss bystanders entering from the right. Mitigation: persist `stereo_format: true` on the manifest entry and let operators override per-dataset.

**Option B**: **VLM-side stereo check** — Extend the Moondream verification prompt to first ask "Is this image a side-by-side stereoscopic capture where the left and right halves show the same scene?" before the multi-person YES/NO check; if yes, drop the video.
  - *Pros*: No coordinate-system changes; works even if the stereo format isn't exactly 2:1.
  - *Cons*: Adds a Moondream call to every pose-positive frame (≈30% wall-time tax on the 90 min of total VLM time observed in the May 17 run); VLM stereo-recognition reliability is unverified — Moondream may not distinguish a wide-FOV single image from a stereo pair.

**Option C**: **Phash-based duplicate-half detector** — Compute a perceptual hash of `frame[:, :w//2]` and `frame[:, w//2:]` and reject if hamming distance < 8.
  - *Pros*: Provably catches stereo without an aspect-ratio assumption; cheap (~1 ms per frame).
  - *Cons*: New dependency (`imagehash`); two stereo halves have slight parallax offset, so the hash threshold needs tuning per camera rig.

Your selection: _____

---

### Issue 3: Wearer's own chin/face at frame bottom passes both YOLO-pose and the VLM gate
**Status**: ⚠️ Confirmed Unresolved — The lowest-scoring passing video, `0030b1e9-c6a6-4809-a495-8d45791f9775` (score `14.13`, 18 frame-detections across 3 tracked "bystanders"), is a single person trimming a hedge alone outdoors with a head-mounted GoPro. Spot-checked frames at `49 s`, `93 s`, `153 s` (`e2e_reports/2026_05_17/frames/worst_frame_{1,2,3}.jpg`) show the wearer's own chin / lower face peeking into the bottom edge of the frame as they look down at their hands. YOLO-pose detects this lower-jaw + collar region as a separate person (confidence 0.61–0.88 across 18 frames), and the Moondream gate confirms it because there *is* technically a human face visible — Moondream cannot tell from the cropped patch that the face it sees belongs to the camera wearer. The Geometric Anti-Wearer Heuristic in `social_presence.py` (drops detections touching the bottom edge without a head/torso) misses this case because the chin patch *does* contain head keypoints (mouth, jaw) and the bbox does not extend to the literal bottom pixel row.

**Option A (recommended)**: **Extend the Anti-Wearer Heuristic to reject "head-only at bottom edge" patches** — Add a rule that drops detections where (a) the bbox's bottom edge is within the bottom 15% of the frame AND (b) no shoulder keypoints are visible AND (c) the bbox height is < 25% of frame height. This is the geometric signature of a chin-in-frame self-detection: a head with no body, low in the frame.
  - *Pros*: Pure-geometry rule, no new models, no VLM cost; the three conditions together are specific enough to spare legitimate bystanders whose faces peek into the bottom edge but who also expose shoulders or a larger body region; complements rather than replaces the existing bottom-edge rule.
  - *Cons*: Could miss a real bystander who is bending down at the bottom of the frame with their head only visible; thresholds (15%, 25%) will likely need tuning on a labeled validation set.

**Option B**: **Gaze-direction VLM check** — When a detection lies in the bottom 20% of the frame, ask Moondream "Is this person's face oriented upward toward the camera (which would indicate the camera wearer's own face peeking into the frame from below)?" — drop if yes.
  - *Pros*: Robust to any geometric pose; uses the VLM that's already loaded.
  - *Cons*: Adds another Moondream call per suspicious frame on top of the existing verification calls; Moondream's gaze-direction reliability is unverified and likely noisy on small face patches.

**Option C**: **Persistent-bottom-edge temporal filter** — Track detections across the 1 FPS sample stream and drop any track that stays anchored within the bottom 20% of the frame for ≥80% of its appearances (the camera-wearer's chin stays in roughly the same place across the whole capture, whereas real bystanders move through the frame).
  - *Pros*: Temporal evidence is harder to spoof than a single-frame heuristic.
  - *Cons*: Requires changing the per-track aggregation in `social_presence.py:detect()`; only kicks in after enough frames have accumulated, so short clips still slip through.

Your selection: _____

---

### Issue 4: Filter pass-rate on first-100 Ego4D set drops to 3% with VLM verify on
**Status**: ⚠️ Confirmed Unresolved — In the May 17 100-video Ego4D E2E run, only 3 / 100 videos passed the social-presence filter (and a spot-check shows 2 of those 3 are false positives — see Issues #2 and #3). The May 16 run on the mixed 80-Charades + 20-Ego4D registry reported 62 / 86 passing (72%) with VLM verify *disabled*. The drop is partly explained by the change in dataset composition (Ego4D `full_scale` files are unscripted egocentric activity, predominantly solo) and partly by the tightened Moondream prompt from Resolved Issue #1 doing its job (correctly rejecting the hotel-TV class), but a 3% pass rate is operationally unworkable — at this yield, building a 10k-video social-interaction corpus requires filtering ~330k Ego4D videos through a ~108 s/video pipeline, which is ≈10 host-days on a Mac Studio. Independently verified: of the 97 rejected videos, the 3 longest-running (e.g. `011ee98a-…`, 9314 s duration, 795 s YOLO+VLM time) were spot-checked and confirmed true negatives (single-person paperwork, hedge-trimming, etc.), so the gate is not pathologically over-rejecting — Ego4D's first-100 slice really is dominated by solo activity.

**Option A (recommended)**: **Pre-filter on Ego4D metadata's `num_persons` scenario tag before YOLO** — `ego4d.json` carries scenario annotations that include rough person-count hints. Skip the entire YOLO+VLM pass for any video whose metadata explicitly tags it as "solo" / "individual" / `num_persons=1`, and reserve the expensive multimodal pass for videos the metadata flags as social or ambiguous.
  - *Pros*: Cuts the per-video cost from 108 s to ~0 s for the obviously-solo majority; raises the *effective* pass rate of the YOLO+VLM stage by removing the easy negatives; uses metadata that's already on disk.
  - *Cons*: Trusts the Ego4D annotation pipeline (which is incomplete — many videos lack person-count tags); requires writing a small `ego4d_metadata_prefilter.py` that joins on `video_uid → scenarios → person_count`; potentially drops the small number of metadata-mislabeled solo→social videos.

**Option B**: **Sample a smaller per-video frame budget when the first N frames are all solo** — In `SocialPresenceDetector.detect()`, if the first ~30 sampled frames produce zero VLM-confirmed bystanders, early-exit the video as "solo" rather than continuing for the full duration.
  - *Pros*: Caps per-video wall-time at ~30 s regardless of total duration; recovers most of the 13-min worst-case observed today.
  - *Cons*: Misses bystanders that only appear in the back half of a long video (e.g. someone walks into the kitchen at the 8-min mark); needs a "minimum coverage" check to avoid false negatives.

**Option C**: **Accept the 3% rate as ground truth and expand the Ego4D download set** — Don't change the filter; instead acquire more raw Ego4D entries up-front so the surviving 3% yields a usable corpus.
  - *Pros*: Filter quality is preserved; aligns with the "weed out unusable data" mandate of this module's Objective.
  - *Cons*: 30× more SSD storage; the Extreme SSD is already at 60 GB for 100 videos, so a 10k corpus implies ~6 TB of raw Ego4D before filtering.

Your selection: _____
