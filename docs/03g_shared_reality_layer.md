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

### 2. Tracking the POV Pan (Saliency / Optical Flow)
Immediately following the task climax (the start of the `task_reaction_window_sec`), we monitor the macroscopic movement of the camera.
- Using simple Optical Flow (or checking the `bystander_detections` relative coordinates), we watch the panning vector of the frame center.

### 3. The Social Reference Logic
- **Condition A (No Ref)**: The camera remains pointed down at the task (the tomato). The POV wearer is confident and does not care about the social boundary.
- **Condition B (Social Referencing)**: The camera explicitly pans away from the task centroid and centers the bystander's bounding box directly into the middle 30% of the screen. The POV actor is "checking in" with the room.
- *Note:* If the Ego4D dataset contains raw eye-tracking telemetry (often included in Aria glasses datasets), check the `gaze_x, gaze_y` arrays to see if the eye-gaze vector snapped to the bystander's face box.

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
- **Batch Test**: During large-scale aggregation, ensure that videos flagged with `social_reference_sought: true` legitimately have a shifting bounding-box origin within the array. Memory constraints are minimal here, but loop efficiency across large coordinate arrays should be benchmarked on the **Mac mini M4 Pro**.
