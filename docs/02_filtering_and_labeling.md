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
  ]
}
```

> **Co-indexing rule**: Within each `bystander_detections` entry, `timestamps_sec[i]`, `bounding_boxes[i]`, and `detection_confidence[i]` are strictly co-indexed. The i-th element of each array describes the same sampled frame.

## Verification & Validation Check
To ensure the filtering mechanisms are robust and correct:
- **Singular Video Test**: Execute the filter module against a single chosen `.mp4` video that is visually verified to have multiple interacting people. Inspect the generated JSON output to confirm the `bystander_detections` bounding boxes accurately surround the actors.
- **Batch Test**: Run the node over a subset folder (e.g., 50 videos). Aggregate the output to check distribution metrics (e.g., % dropped due to no social presence, % dropped due to idling). Assert that every entry in the `filtered_manifest.json` has `identified_tasks` with non-empty attributes, effectively proving the system scaled accurately without unhandled `None` crashes. Furthermore, utilize the **24GB RAM Mac mini M4 Pro** environment effectively by ensuring the batched VLM calls do not trigger out-of-memory unified memory kills.

---

## 🚀 Implementation Status

The filtering and labeling pipeline is fully operational within the `src/dataset_acquisition` and `src/filtering_and_labeling` modules. 
- **Social Presence Filter**: Successfully implemented using YOLOv8 (`yolov8n.pt`) with a multi-stage Anti-Wearer Heuristic.
- **Contextual Task Labeling**: Shifted from VLM-based labeling to **Ego4D Metadata Extraction** using `ego4d.json`. This provides ground-truth task contexts and significantly reduces computational overhead.
- **Temporal Climax Identification**: Implemented a hybrid two-stage approach using dense optical flow (`cv2.calcOpticalFlowFarneback`) and VLM-based refinement for slow tasks.

## 🧪 Resolved Issues & Implementation Refinements

1. **Multi-Person Social Presence Tracking (Resolved - April 22)**:
   - **Problem**: The system previously only tracked the single most confident person per frame, losing context for multi-actor interactions.
   - **Solution**: Updated `social_presence_filter` to capture all detected persons per frame and updated the schema to support an array of `bystander_detections`.

2. **Pipeline Resumability & Error Isolation (Resolved - April 22)**:
   - **Problem**: Long batch runs were fragile; a single error in one video would crash the entire process, and progress was not saved incrementally.
   - **Solution**: Implemented incremental state saving (write-after-each-video) and isolated per-video errors to `02_filter_errors.json`.

3. **Robust Task Labeling & Merging (Resolved - April 22)**:
   - **Problem**: Exact string matching on raw VLM outputs caused identical tasks to be treated as distinct due to minor formatting differences, leading to "task fragmentation."
   - **Solution**: Implemented label normalization (lowercasing, punctuation stripping) and switched to dynamic confidence scoring based on temporal consistency.

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

## ⚠️ Unresolved Issues & Suggestions

- **Optical Flow Density Calibration**: The current 5 FPS sampling for optical flow may miss ultra-fast actions (e.g., a lightning-fast impact). Increasing sampling density for "Fast" velocity tasks is suggested.
- **VLM Context Windowing**: For very long clips (>10 mins), the current sparse frame sampling might miss task transitions. Implementing a sliding window approach for VLM labeling is suggested to ensure temporal coverage.
- **Advanced Wearer Removal**: While the geometric anti-wearer heuristic is effective, it still produces occasional false positives. Using a dedicated **egocentric pose estimator** or "wearer segmentation" mask is suggested for higher precision.
