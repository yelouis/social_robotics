# AI Task Breakdown: Shared Reality Layer (03g)

## Objective
The **Shared Reality Layer** mathematically evaluates the psychological concept of "Social Referencing." In classic tests (like the Visual Cliff experiment), an infant facing an ambiguous or new situation pauses, turns to the caregiver, and seeks validation. This layer tracks the camera telemetry to see if the POV actor explicitly *verifies* the bystander's reaction after completing a task.

---

## 📥 Input Requirements
- **`filtered_manifest.json`**: For task tracking and `bystander_detections` bounding boxes.
- **Raw Video / Saliency Engine**: To track what the camera is focusing on.

---

## 🛠️ Implementation Strategy

### 1. Identifying the Task Centroid
Use Ego4D task bounding boxes (or center-frame assumption during the `task_climax_sec`) to identify the visual coordinates of the action being performed (e.g., cutting a tomato on the counter). 

### 2. Tracking the POV Pan (Optical Flow)
Immediately following the task climax (the start of the `task_reaction_window_sec`), we monitor the macroscopic movement of the camera using Farneback optical flow.
- **Bystander Masking**: Before calculating the mean optical flow of the background, the bounding boxes of any bystanders are dynamically masked out. This prevents foreground bystander motion from contaminating the egocentric camera shift estimate.
- **Adaptive Resolution & Sampling**: To maintain performance, optical flow is temporally subsampled to a target of 10 FPS (`TARGET_FLOW_FPS = 10.0`). Spatial resolution is dynamically tiered based on host memory: 1.0x (full resolution) for hosts with >= 48 GB unified memory to capture sub-degree micro-pans, and 0.5x downsampled for smaller hosts.

### 3. The Social Reference Logic
- **Condition A (No Ref)**: The camera remains pointed down at the task. The POV wearer is confident and does not care about the social boundary.
- **Condition B (Social Referencing)**: The system registers a social referencing event only if both of the following conditions are met:
  1. **Significant Camera Pan**: The camera explicitly pans away from the task centroid, defined as an accumulated optical flow shift magnitude exceeding 4% of the frame's diagonal (`SHIFT_THRESHOLD_RATIO = 0.04`). This dynamic threshold ensures scale invariance across different clip resolutions.
  2. **Temporal Bystander Centering**: The bystander's bounding box must land within the central 35%–65% region of the screen explicitly during the *final 25%* of the reaction window (`FINAL_CENTERING_TAIL_FRACTION = 0.25`). This temporal ordering constraint guarantees the pan *resulted* in the bystander being centered, eliminating false positives from coincidental pre-centering.

### 4. Deferred Alternative: Aria Glasses Gaze Telemetry Integration
While extracting sub-degree gaze telemetry (`gaze_x`, `gaze_y`) from Aria glasses recordings (`.vrs` formats via `projectaria_tools`) would provide high-precision ground-truth tracking for social referencing, this approach has been intentionally deferred. The Aria-glasses subset of the project's Ego4D corpus is too small to justify the heavy (~200 MB) `projectaria_tools` dependency, the per-frame `.vrs` parsing cost, and the complexity of a dual-signal architecture. The pipeline relies uniformly on the optical flow and bounding box centering heuristic across all clips.

---

## 📤 Output Schema and Integration
**Example Output Data (`03g_shared_reality_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03g_shared_reality",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "post_climax_camera_shift_vector": [450, -220],
      "bystander_centered_in_fov": true,
      "social_reference_sought": true
    }
  ]
}
```

## Verification & Validation Check
- **Singular Video Test**: Pick an ambiguous Ego4D clip where someone builds something and then looks up to show a friend. Print out the `camera_shift_vector` timeline. Assert that the `bystander_centered_in_fov` triggers true concurrently with the frame visually aligning the bystander's face in the center matrix.
- **Batch Test**: During large-scale aggregation, ensure that videos flagged with `social_reference_sought: true` legitimately have a shifting bounding-box origin within the array. Memory constraints are minimal here, but loop efficiency across large coordinate arrays should be benchmarked on the **Mac Studio (M4 Max, 64 GB unified memory)**.

---

## 🧪 Resolved Issues & Implementation Refinements

*(All historical issues have been integrated into the current pipeline architecture.)*


## ⚠️ Unresolved Issues & Suggestions

*(No unresolved issues at this time.)*
