# AI Task Breakdown: Video Filtering and Labeling

## Objective
To ensure computational efficiency and relevance, not all downloaded videos make it to the advanced Social Feature Layers. This module performs two critical filtration passes to weed out unusable data from first-person (POV) perspectives.

---

## 1. The Social Presence Filter
Since our project relies on investigating social-affective interactions, a video of a person entirely alone is useless. 

**Criteria**: 
The video **must** contain more than one person. Since this is an egocentric/POV video, the person whose perspective is being shown (the camera wearer) *does not count* towards this total. There must be at least one other visible human actor present in the frame.

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
We must actively generate task labels spanning the duration of the clip. If no clear tasks are taking place at any point, the video is dropped.

**Task Requirements**:
- Write a VLM processing layer that observes interval frames from the video to classify the sequential actions taking place.
- Generate an array of `identified_tasks`.
- *Drop Rule*: If the VLM states "Idling," "No clear task," or "Ambiguous activity" for the entire video duration, it must be discarded.

---

## 3. Temporal Task Climax Identification
Knowing *what* the task is isn't enough; downstream affective layers need to know exactly *when* it happens so they can measure bystander reactions accurately.

**Criteria**:
Identify the exact timestamp or narrow temporal window where the "climax" or core action of each task occurs (e.g., the exact moment the dropped glass shatters, or the exact moment the basketball leaves the hands).

**Task Requirements**:
- **Hybrid Optical Flow + VLM Climax Detection**: A two-stage approach that requires zero additional model downloads:
  - **Stage 1 — Optical Flow Variance Peak**: Use OpenCV's `cv2.calcOpticalFlowFarneback` to compute dense optical flow at ~5 FPS within each task's temporal bounds. The frame with the highest flow magnitude is the kinetic `task_climax_sec`. This is highly accurate for abrupt physical actions (dropping, slipping, throwing).
  - **Stage 2 — VLM Refinement (slow/cognitive tasks)**: For tasks classified as `"slow"` velocity (e.g., solving a puzzle, reading a sign), sample 3-5 candidate frames around the optical flow peak and use the existing **moondream** VLM to score each frame's proximity to the action climax, using the `task_label` as context.
- **Dynamic Action Velocity & Delay Buffers**: Human reactions vary wildly depending on the abruptness of a task. Node 02's VLM will classify the "Velocity" of the task. We use this to compute a dynamic `task_reaction_window_sec`:
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

## Implementation Status & Findings

### Accomplished
- **Social Presence Filter**: Successfully implemented using YOLOv8 (`yolov8n.pt`) via the `ultralytics` package. Downsampled frames (1 FPS) are scanned for 'person' classes. Output properly structures the strictly co-indexed bounding boxes, timestamps, and confidence scores.
- **Contextual Task Labeling**: Implemented the frame sampling logic and integration with `ollama` Python client to query a local VLM. 
- **Temporal Task Climax Identification**: Successfully implemented `cv2.calcOpticalFlowFarneback` to compute dense optical flow at ~5 FPS within the identified task segment. The kinetic peak magnitude and dynamic `task_reaction_window_sec` are accurately captured.
- **Schema Validation**: The pipeline accurately outputs `filtered_manifest.json` completely aligned with the defined schema, correctly dropping videos when no tasks or no bystanders are present.

### Encountered Problems
1. **Moondream Instruction Following (Resolved)**: The initial use of `moondream` (1.8B) resulted in poor instruction following for complex, multi-part prompt formatting. It frequently returned empty strings or hallucinatory text.
2. **Empty Task Extraction (Resolved)**: Using the lightweight model caused valid social interactions to be dropped due to inaccurate scene parsing.

### Current Implementation & Solutions
- **Model Upgrade (Implemented)**: The pipeline has been upgraded to use **Qwen2.5-VL** via Ollama for the Contextual Task Labeling phase. This model possesses significantly stronger instruction-following capabilities and visual understanding, resolving the issues with structured output formatting and scene parsing accuracy.
- **Robust Prompting**: Even with a stronger model, the pipeline maintains a robust fallback logic to categorize tasks as "Idling" if the VLM explicitly indicates no activity, ensuring only high-quality social data is persisted.

## 🚀 Resolved Issues & Pipeline Hardening (April 2026)

Following a comprehensive audit, the following critical refinements were implemented to ensure the pipeline is production-ready:

1. **Multi-Person Social Presence Tracking (Resolved)**:
   - **Problem**: The system previously only tracked the single most confident person per frame.
   - **Solution**: The `social_presence_filter` now leverages the shared `SocialPresenceDetector` to capture **all** detected persons per frame. The output schema has been updated to return an array of `bystander_detections`, each with its own `person_id`, ensuring downstream layers can perform multi-actor social analysis.

2. **Pipeline Resumability & Error Isolation (Resolved)**:
   - **Problem**: Long runs were fragile and non-resumable.
   - **Solution**: Implemented incremental state saving (write-after-each-video) and a processed-ID skip check. Processing errors are now caught and isolated to `02_filter_errors.json`, preventing a single corrupt video from crashing the entire batch.

3. **Robust Task Labeling & Merging (Resolved)**:
   - **Problem**: Exact string matching on VLM output caused task fragmentation.
   - **Solution**: Implemented label normalization (lowercasing, stripping punctuation) during the merging phase. Additionally, `task_confidence` is now dynamically calculated based on the temporal consistency (merge count) of the task rather than being a hardcoded value.

4. **Stage 2 VLM Climax Refinement (Resolved)**:
   - **Problem**: Optical flow alone was insufficient for slow/cognitive tasks.
   - **Solution**: Implemented the documented two-stage climax identification. For tasks with `slow` velocity, the pipeline now samples candidate frames and uses **Qwen2.5-VL** to score and refine the climax timestamp.

5. **Schema Consistency & Temporal Bounds (Resolved)**:
   - **Problem**: Task start/end times were missing from the final manifest.
   - **Solution**: Formally promoted `task_start_sec` and `task_end_sec` to the official output schema. This provides critical temporal context for downstream social layers like Reasonable Emotion (03b).

6. **Infrastructure & Robustness (Resolved)**:
   - **Problem**: Fragile temp directories and stale documentation.
   - **Solution**: Refactored to use `tempfile.TemporaryDirectory` for safe frame extraction. Updated all docstrings to reflect the upgrade to Qwen2.5-VL. Fixed `run_verification.py` to ensure manifests are written to the correct SSD output paths.
