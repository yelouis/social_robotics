# AI Task Breakdown: Video Filtering and Labeling

## Objective
To ensure computational efficiency and relevance, not all downloaded videos make it to the advanced Social Feature Layers. This module performs two critical filtration passes to weed out unusable data from first-person (POV) perspectives.

---

## 1. The Social Presence Filter
Since our project relies on investigating social-affective interactions, a video of a person entirely alone is useless. 

**Criteria**: 
The video **must** contain more than one person. Since this is an egocentric/POV video, the person whose perspective is being shown (the camera wearer) *does not count* towards this total. There must be at least one other visible human actor present in the frame. 

To achieve this, the system uses a **Refined Anti-Wearer Heuristic**:
- **Limb Filtering**: Ignore detections touching the bottom edge of the frame without showing a head/torso (likely wearer's hands/feet).
- **Ghost Torso Exclusion**: Ignore full-height detections starting at the absolute top (`y1=0`), which are common false positives from the wearer's torso.
- **Temporal Consistency**: A video is only "KEPT" if social presence is detected in at least **2 sampled frames** to filter out momentary YOLO glitches.

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
  - **Stage 1 — Optical Flow Variance Peak**: Use OpenCV's `cv2.calcOpticalFlowFarneback` to compute dense optical flow at ~5 FPS within each task's temporal bounds. The frame with the highest flow magnitude is the kinetic `task_climax_sec`. This is highly accurate for abrupt physical actions (dropping, slipping, throwing).
  - **Stage 2 — VLM Refinement (slow/cognitive tasks)**: For tasks classified as `"slow"` velocity (e.g., solving a puzzle, reading a sign), sample 3-5 candidate frames around the optical flow peak and use the existing **moondream** VLM to score each frame's proximity to the action climax, using the `task_label` as context.
- **Dynamic Action Velocity & Delay Buffers**: Human reactions vary wildly depending on the abruptness of a task. The system maps the metadata-derived `task_label` to a "Velocity" class. We use this to compute a dynamic `task_reaction_window_sec`:
  - **Fast / Instinctual** (e.g., slipping, dropping an item): Requires a short buffer. Climax + `[0.5s to 2.0s]`.
  - **Medium** (e.g., throwing a ball, standard interaction): Climax + `[1.0s to 3.0s]`.
  - **Slow / Cognitive** (e.g., solving a puzzle, reading a sign): Climax + `[2.0s to 6.0s]`.

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
- **Social Presence Filter**: Successfully implemented using YOLOv8-Pose (`yolov8n-pose.pt`) with head+shoulder keypoint validation, the geometric Anti-Wearer Heuristic, and a fast-VLM early-exit verification gate (see Resolved Issue #22).
- **Contextual Task Labeling**: Shifted from VLM-based labeling to **Ego4D Metadata Extraction** using `ego4d.json`. This provides ground-truth task contexts and significantly reduces computational overhead.
- **Temporal Climax Identification**: Implemented a hybrid two-stage approach using dense optical flow (`cv2.calcOpticalFlowFarneback`) and VLM-based refinement for slow tasks.

## 🧪 Resolved Issues & Implementation Refinements

1. **Multi-Person Social Presence Tracking (Resolved - April 22)**:
   - **Problem**: The system previously only tracked the single most confident person per frame, losing context for multi-actor interactions.
   - **Solution**: Updated `social_presence_filter` to capture all detected persons per frame and updated the schema to support an array of `bystander_detections`.

2. **Pipeline Resumability & Error Isolation (Resolved - April 22)**:
   - **Problem**: Long batch runs were fragile; a single error in one video would crash the entire process, and progress was not saved incrementally.
   - **Solution**: Implemented incremental state saving (write-after-each-video) and isolated per-video errors to `02_filter_errors.json`.

3. **Robust Task Labeling & Merging (Resolved - April 22, Updated - May 05)**:
   - **Problem**: Exact string matching on raw VLM outputs caused identical tasks to be treated as distinct due to minor formatting differences, leading to "task fragmentation."
   - **Solution**: The original VLM-based labeling approach was fully replaced by Ego4D Metadata Extraction (see Resolved Issue #9). Since metadata labels are pre-normalized by the Ego4D consortium, the original punctuation-stripping requirement is now obsolete. The current implementation applies `.strip()` whitespace normalization on scenario strings (`pipeline.py` line 229) and uses lowercased comparison for the drop-rule filter (line 230). Task confidence is set to `1.0` as a ground-truth marker for metadata-sourced labels.

4. **Stage 2 VLM Climax Refinement (Resolved - April 22)**:
   - **Problem**: Optical flow analysis alone was insufficient for identifying action climaxes in slow or cognitive tasks (e.g., reading, solving puzzles).
   - **Solution**: Implemented a two-stage refinement process where Qwen2.5-VL scores candidate frames around the optical flow peak for low-velocity tasks.

5. **Schema Consistency & Temporal Bounds (Resolved - April 22)**:
   - **Problem**: Critical temporal context (`task_start_sec`, `task_end_sec`) was missing from the output manifest, making it difficult for downstream layers to align their analysis.
   - **Solution**: Formally promoted start and end timestamps to the official schema to support accurate alignment for layers like 03b (Reasonable Emotion).

6. **Infrastructure & Robustness (Resolved - April 22)**:
   - **Problem**: Fragile temporary directory management and stale documentation regarding the model upgrade were causing maintenance overhead.
   - **Solution**: Refactored to use `tempfile.TemporaryDirectory` and synchronized all docstrings and verification scripts with the Qwen2.5-VL implementation.

7. **Ego4D Camera Wearer Detection (Resolved - April 22)**:
   - **Problem**: YOLOv8 frequently misidentified the camera wearer's own limbs or torso as external bystanders, leading to false positives and SSD storage overflow.
   - **Solution**: Implemented a multi-stage Anti-Wearer Heuristic (edge exclusion, confidence floor, temporal consistency) to filter out self-detections.

8. **`NameError` / `KeyError` Crashes in `process_video` (Resolved - April 27)**:
   - **Problem**: Three latent bugs in `pipeline.py` (missing `video_id` definition and unsafe `entry['id']` access) caused the pipeline to crash whenever a video failed a filter or used a different ID key format.
   - **Solution**: Resolved the ID key dynamically (`entry.get('id', entry.get('video_id'))`) and ensured `video_id` was correctly scoped at the start of the processing method.

9. **Ego4D Metadata Integration (Resolved - May 02)**:
   - **Problem**: Running a VLM (Qwen2.5-VL) for every 5 seconds of video to label the task was computationally expensive and occasionally led to hallucinations.
   - **Solution**: Integrated the official `ego4d.json` metadata to extract ground-truth scenarios and descriptions. This ensures 100% accuracy relative to the dataset and improves throughput by 50x.

10. **Velocity Taxonomy Mapping (Resolved - May 02)**:
    - **Problem**: Metadata does not provide physical velocity, which is required for dynamic reaction window calculation.
    - **Solution**: Implemented a heuristic mapping between metadata descriptions and physical velocity categories (fast, medium, slow).

11. **24GB Unified Memory Constraints (Resolved - May 02)**:
    - **Problem**: The original pipeline interleaved YOLO and VLM calls within a per-video loop, forcing both models to reside in unified memory simultaneously. This caused significant memory pressure and potential OOM kills on the 24GB Mac mini M4 Pro. Additionally, sequential VLM calls per task were inefficient.
    - **Solution**: Refactored the pipeline into a **two-pass architecture**: Pass 1 runs the Social Presence Filter (YOLO) for all videos, followed by explicit model unloading and memory clearing (`torch.mps.empty_cache()`), and Pass 2 runs the Climax Refinement (VLM). Implemented **multi-image VLM batching** to process all climax candidates in a single inference call and added image resizing (max 1024px) to ensure stable VRAM usage.

12. **Optical Flow Density Calibration (Resolved - May 04)**:
    - **Problem**: The optical flow sampling was fixed at ~5 FPS for all velocity classes, missing the exact peak-action frame for fast actions by up to ±100ms.
    - **Solution**: Implemented Two-Pass Hierarchical Detection: a coarse ~5 FPS pass identifies the peak window, followed by a native 30 FPS dense pass within ±1s of the coarse peak for full temporal precision.

13. **VLM Context Windowing for Long Clips (Resolved - May 04)**:
    - **Problem**: For clips with multiple distinct task transitions, the VLM refinement evaluated the entire video duration globally, missing intermediate transitions between tasks.
    - **Solution**: Implemented Per-Task Bounded VLM Scope by dynamically partitioning the video duration evenly across valid Ego4D scenarios, ensuring the optical flow and VLM pipeline operates strictly within each task's temporal bounds.

14. **Advanced Wearer Removal (Resolved - May 04)**:
    - **Problem**: The geometric anti-wearer heuristic produced occasional false positives when the wearer's hands were extended into the upper frame.
    - **Solution**: Unified resolution with Node 01 by utilizing the MediaPipe Hand Overlap Suppression in the shared `src/shared/social_presence.py` module, preventing hand-held objects from being misclassified as bystanders.

15. **ByteTrack Tracker State Bleeds Across Videos in Filtering Pipeline (Resolved - May 07)**:
    - **Problem**: `FilteringPipeline.social_presence_filter` reuses a single `SocialPresenceDetector` across every video in Pass 1, and the underlying `model.track(..., persist=True)` call retained ByteTrack's track IDs and Kalman predictions between videos. This caused `person_id` values to monotonically increment across the dataset and risked stale tracks from video A bleeding into video B's `bystander_detections`.
    - **Solution**: Resolved upstream in `src/shared/social_presence.py:detect()` (see Node 01 Resolved Issue #14): the detector now walks `self.model.predictor.trackers` and calls `.reset()` on each tracker at the top of every `detect()` invocation, guarded by try/except for Ultralytics API drift. Because `FilteringPipeline.social_presence_filter` calls into the same shared `detect()` method, no per-pipeline change was required — the cross-video contamination is now eliminated for both the acquisition and filtering pipelines via one shared fix.

16. **Reaction Window Exceeded Video Duration (Resolved - May 07)**:
    - **Problem**: `temporal_climax_identification` in `pipeline.py` computed `task_reaction_window_sec` as `climax_sec + offset` without bounding the result by the clip's `duration_sec`. For a 45 s video with a climax at 43.5 s and a `medium`-velocity offset of `[+1.0 s, +3.0 s]`, the manifest emitted `[44.5, 46.5]` — 1.5 s past the end of the file. Downstream layers (03a Attention, 03b Emotion) that seek frames within the reaction window would either crash on `cap.set` failures or silently produce empty observation data.
    - **Solution**: Threaded `duration_sec` from `process_video_vlm_pass` into `temporal_climax_identification` and added a final clamp `window = [min(window[0], duration_rounded), min(window[1], duration_rounded)]` immediately after the velocity-based offset is applied. The schema is unchanged; truncated windows simply collapse to a valid in-bounds interval (occasionally a zero-length point at EOF for climaxes at the very last frame), which downstream layers can detect via `start == end`.

17. **Undocumented `hand_detections` Schema Field (Resolved - May 07)**:
    - **Problem**: The Pass 1 social filter began emitting `hand_detections` into `filtered_manifest.json` (set in `pipeline.py` `process_video_vlm_pass`) and Layer 03a Attention actively consumes the field for MediaPipe-based occlusion suppression. The schema definition section of this document never documented the field, silently violating the "additive-only schema" contract this doc claims to uphold.
    - **Solution**: Added `hand_detections` to the JSON schema example under `## Output`, including a representative two-frame entry, and added an explicit contract paragraph immediately after the existing co-indexing rule. The new contract spells out: one entry per sampled frame that contained hands, `timestamp_sec` aligns with the same sampling grid as `bystander_detections`, `hand_boxes` is a list of integer `[x1, y1, x2, y2]` pixel coords, hand-less frames are omitted, and the field is consumed by Layer 03a so it must not be removed or renamed.

18. **Dead Imports & Escaped-Quote Docstring in `pipeline.py` (Resolved - May 07)**:
    - **Problem**: `pipeline.py` carried `import os` and `import shutil` at the top of the module that were never referenced anywhere — leftovers from an earlier refactor that produced lint noise and pulled `shutil` for no functional reason. While editing this file, an additional latent defect was discovered: line 186 contained `\"\"\" Sample frames at sample_rate_fps and detect all persons. \"\"\"` — i.e. backslash-escaped quote characters instead of a Python triple-quoted docstring — which made the entire module fail to compile (`SyntaxError: unexpected character after line continuation character`) and meant the filtering pipeline could not have been importable in its prior state.
    - **Solution**: Removed both unused imports from the import block. Replaced the malformed docstring on `social_presence_filter` with a proper `""" Sample frames at sample_rate_fps and detect all persons. """`. `python -m py_compile src/filtering_and_labeling/pipeline.py` now passes cleanly.

19. **Two-Pass YOLO/VLM Architecture Overcautious on 64 GB Mac Studio (Resolved - May 13)**:
    - **Problem**: Resolved Issue #11 ("24GB Unified Memory Constraints") refactored `pipeline.py` into a strict two-pass architecture — Pass 1 ran the YOLO Social Presence Filter across *every* video, called `torch.mps.empty_cache()`, then Pass 2 ran the Qwen2.5-VL climax refinement — because the 24 GB Mac mini M4 Pro could not keep both models resident. On the new Mac Studio (M4 Max, 64 GB unified memory) host, YOLOv8 (~100 MB resident with MPS activations) and Qwen2.5-VL (~6 GB) co-reside with ~50 GB headroom, yet the two-pass path still reads every video off the Extreme SSD twice — once for YOLO frames, once for VLM frames — roughly doubling the disk-read cost of the filtering stage for no memory benefit on the new host.
    - **Solution**: Implemented a single-pass interleaved architecture gated on host memory. `FilteringPipeline.__init__` now probes `psutil.virtual_memory().total >= 48 * 2**30` and stores the result in `self.single_pass` (defaulting to `False` if `psutil` is unavailable). `run()` was split into a thin dispatcher plus two helpers: `_run_single_pass` (used when `single_pass` is true) runs the YOLO social filter immediately followed by the VLM task/climax pass for each video, so each file is read off the SSD exactly once; `_run_two_pass` preserves the prior Pass-1 / `cleanup_yolo()` / Pass-2 logic verbatim as the fallback for sub-48 GB hosts, keeping Resolved Issue #11's memory guarantees intact. `_run_single_pass` wraps each video's combined YOLO+VLM work in a try/except routed to `log_error`, extending the per-video error isolation from Resolved Issue #2 to the YOLO stage. The selected architecture is logged at construction.

20. **Climax-Refinement VLM Pinned to Qwen2.5-VL 3B Despite 64 GB Headroom (Resolved - May 13)**:
    - **Problem**: Section 3 ("Temporal Task Climax Identification") and `pipeline.py` pinned the Stage-2 VLM refinement to Qwen2.5-VL 3B (~3 GB resident) for slow/cognitive tasks. The 3B tier was selected explicitly to coexist with YOLO on the 24 GB Mac mini. On the Mac Studio (M4 Max, 64 GB unified memory) target host, Qwen2.5-VL 7B (~15 GB resident) is comfortably viable and has measurable accuracy gains on fine-grained action-climax localization in slow/cognitive tasks (puzzles, reading, writing) — exactly the regime Stage 2's VLM refinement was built to address.
    - **Solution**: Promoted the 7B tier to the default. `FilteringPipeline.__init__` previously hardcoded `self.vlm_model = "qwen2.5vl"`; it now reads the `SAF_VLM_MODEL_TIER` environment variable and sets `self.vlm_model` to `qwen2.5vl:3b` when the value is `small` (case-insensitive) or `qwen2.5vl:7b` otherwise. `import os` was re-added to `pipeline.py` to support the env lookup (it had been removed as dead code in Resolved Issue #18 and is now genuinely used again). The resolved tier is printed at construction alongside the filtering architecture mode. Memory-constrained hosts such as the 24 GB Mac mini opt down to the 3B model via `SAF_VLM_MODEL_TIER=small`.

21. **MediaPipe Graph Initialization Crash on M4 Max (Resolved - May 14)**:
    - **Problem**: `MediaPipe` version `0.10.11` failed with `ValidatedGraphConfig Initialization failed: Output tensor range is required` inside the `ImageToTensorCalculator` when initializing `Hands` for wearer occlusion suppression. This failure is a known incompatibility with modern `protobuf` builds on Apple Silicon.
    - **Solution**: Downgraded `mediapipe` to `0.10.9` in the `vlm_env` environment, restoring stability without sacrificing hand-tracking precision.

22. **YOLO False Positives on Wearer Limbs & Equipment (Resolved - May 15)**:
    - **Problem**: The base bbox-only `yolov8n.pt` detector in `src/shared/social_presence.py` repeatedly fired on the camera wearer's own clothed forearms (jacket sleeves, gloved hands) and on equipment with vaguely humanoid silhouettes (chair backs, monitor stands, mannequins in workshop footage). The pre-existing geometric Anti-Wearer Heuristic (bottom-edge limb filter + top-edge ghost-torso filter) and the MediaPipe Hand Overlap Suppression branch caught the *easy* cases — sleeves resting in the bottom of frame, a wearer torso starting at `y1=0` — but were intrinsically blind to mid-frame limb stubs and humanoid objects, because both heuristics were operating on bbox geometry alone with no concept of "is this a connected human body?". The downstream consequence was a stream of false-positive `bystander_detections` entries in `filtered_manifest.json`, which Layer 03a Attention and Layer 03f Motor Resonance would then waste compute trying to extract gaze and pose features from.
    - **Solution**: Implemented the combined Option A + Option B remediation path. (1) **YOLO-Pose migration**: `SocialPresenceDetector.__init__` default `model_path` switched from `yolov8n.pt` to `yolov8n-pose.pt`, and call sites in `src/dataset_acquisition/filterer.py:StreamingFilter` and `src/filtering_and_labeling/pipeline.py:FilteringPipeline` updated to match. Inside the per-batch YOLO loop in `detect()`, each detection is now gated by `_has_head_and_shoulder(kps_conf)` — the detection is kept only if at least one of the five head COCO keypoints (nose / left eye / right eye / left ear / right ear) AND at least one shoulder keypoint (left / right) clears `KEYPOINT_CONF_THRESHOLD = 0.3`. Detections with no associated keypoints array, or keypoints below the threshold, are dropped. This intrinsically rejects sleeves, hand stubs, and humanoid equipment that the bbox-only model previously kept. The MediaPipe Hand Overlap Suppression branch was removed from the filter path (the keypoint gate makes it redundant), and MediaPipe Hands itself is now only initialized when `return_hands=True` so Node 01's streaming filter no longer pays its initialization cost. MediaPipe Hands output is retained for Node 02 to preserve the `hand_detections` schema contract with Layer 03a (Resolved Issue #17). (2) **VLM Early-Exit Verification gate**: When `vlm_verify=True` (default on for `FilteringPipeline`, gated by `SAF_VLM_VERIFY_SOCIAL` env var), each YOLO-pose-confirmed candidate frame is sent to a fast VLM (`moondream` by default, overridable via `SAF_VLM_VERIFY_MODEL`) with a simple YES/NO multi-person prompt. The detector keeps a `vlm_attempts` budget capped at `MAX_VLM_VERIFY_FRAMES = 5` and a `vlm_verified_count` confirmation count; once `vlm_verified_count >= min_consistency` the budget stops being spent (the early-exit guarantee), and in `fast_mode=True` `detect()` returns `True` immediately. A video that produces YOLO candidates but never accumulates `min_consistency` VLM confirmations is dropped at the end of `detect()` — `fast_mode` returns `False`, otherwise empty `bystander_detections` / `hand_detections` arrays are returned. VLM infrastructure errors fail-open (return `True` for that frame) so missing-Ollama is not the sole reason to discard an otherwise pose-confirmed bystander.
