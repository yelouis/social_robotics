# AI Task Breakdown: Video Filtering and Labeling

## Objective
To ensure computational efficiency and relevance, not all downloaded videos make it to the advanced Social Feature Layers. This module performs two critical filtration passes to weed out unusable data from first-person (POV) perspectives.

---

## 1. The Social Presence Filter
Since our project relies on investigating social-affective interactions, a video of a person entirely alone is useless. 

**Criteria**: 
The video **must** contain more than one person. Since this is an egocentric/POV video, the person whose perspective is being shown (the camera wearer) *does not count* towards this total. There must be at least one other visible human actor present in the frame.

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
- **BayesianVSLNet Alignment**: Run a Video Summarization / Highlight Detection model (such as **BayesianVSLNet**) on the clip isolating each identified task. This model analyzes the visual features to find the statistical apex (`task_climax_sec`) of the action.
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
        "climax_extraction_model": "BayesianVSLNet"
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
        "climax_extraction_model": "BayesianVSLNet"
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

## Verification & Validation Check
To ensure the filtering mechanisms are robust and correct:
- **Singular Video Test**: Execute the filter module against a single chosen `.mp4` video that is visually verified to have multiple interacting people. Inspect the generated JSON output to confirm the `bystander_detections` bounding boxes accurately surround the actors.
- **Batch Test**: Run the node over a subset folder (e.g., 50 videos). Aggregate the output to check distribution metrics (e.g., % dropped due to no social presence, % dropped due to idling). Assert that every entry in the `filtered_manifest.json` has `identified_tasks` with non-empty attributes, effectively proving the system scaled accurately without unhandled `None` crashes. Furthermore, utilize the **24GB RAM Mac mini M4 Pro** environment effectively by ensuring the batched VLM calls do not trigger out-of-memory unified memory kills.
