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

### 4. Future Possible Direction: Aria Glasses Gaze Telemetry Integration
The current pipeline relies exclusively on optical flow and bounding box centering. Future iterations may explore extracting sub-degree gaze telemetry (`gaze_x`, `gaze_y`) from Aria glasses recordings (`.vrs` formats via `projectaria_tools`). This would provide high-precision ground-truth tracking for social referencing, although it is limited to the Aria-specific subset of clips.

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

---

## 🧪 Resolved Issues & Implementation Refinements

1. **Inverted Optical Flow Directionality (Resolved - May 04)**:
   - **Problem**: When calculating the camera shift vector, the initial implementation treated the background's optical flow movement directly as the camera's movement. If the camera panned right (+X), the background moved left (-X), resulting in an incorrectly inverted `camera_shift_vector` in the output JSON.
   - **Solution**: Updated the `_extract_camera_shift` method in `pipeline.py` to invert the mean optical flow of the background (`mean_dx = -np.mean(flow[..., 0])`). This accurately estimates the camera's egocentric panning vector based on the static background's relative displacement.

2. **Bystander Centering Logic Scaling (Resolved - May 04)**:
   - **Problem**: The raw bystander bounding boxes lacked dynamic referencing to the source video's resolution, causing the "middle 30%" centering threshold to fail on non-standard Ego4D clip aspect ratios.
   - **Solution**: Refactored the `_check_bystander_centering` function to dynamically extract the `width` and `height` from the `cv2.VideoCapture` properties, establishing a dynamic 35% to 65% bounding box threshold that is resolution-agnostic.

3. **Missing `__init__.py` Module File (Resolved - May 04)**:
   - **Problem**: The `src/layer_03g_shared_reality/` directory was missing an `__init__.py` file, unlike every sibling layer module (e.g., 03f). This caused `ImportError` when attempting to import the pipeline as a Python package rather than via `PYTHONPATH` manipulation.
   - **Solution**: Created `src/layer_03g_shared_reality/__init__.py` with the standard re-export (`from .pipeline import SharedRealityPipeline`), matching the pattern established by `layer_03f_motor_resonance/__init__.py`.

4. **`social_reference_sought` False Positive — No Camera Shift Requirement (Resolved - May 04)**:
   - **Problem**: The `social_reference_sought` flag was set solely based on `bystander_centered_in_fov`, ignoring whether the camera actually panned. According to the specification (Section 3 — "The Social Reference Logic"), social referencing occurs when "the camera explicitly pans away from the task centroid AND centers the bystander." A bystander who was already centered *before* the task climax would produce a false positive, as there was no deliberate gaze shift.
   - **Solution**: Added a `shift_magnitude` check (`np.sqrt(dx² + dy²) > 30.0` pixels) as a conjunction with `bystander_centered_in_fov`. The `social_reference_sought` flag now requires both conditions to be true, matching the documented two-condition social referencing logic.

5. **Missing Bystander Gate in `process_video` (Resolved - May 04)**:
   - **Problem**: Unlike the sibling 03f layer, which gates on `not bystanders or not tasks`, the 03g layer only gated on `not tasks`. Videos with empty `bystander_detections` arrays would still be processed and produce result entries with `bystander_centered_in_fov: false` and `social_reference_sought: false`, polluting the output JSON with meaningless records.
   - **Solution**: Added the missing `not bystanders` check alongside `not tasks` in `process_video()`, causing the pipeline to return `None` early for videos without bystander data.

6. **`np.bool_` Type Leak Into JSON Output (Resolved - May 04)**:
   - **Problem**: The `shift_magnitude > 30.0` comparison returned `np.bool_` (a numpy scalar), not a native Python `bool`. When combined with `bystander_centered` via `and`, the result type was `np.True_` or `np.False_`. This caused `isinstance(value, bool)` checks to fail and would produce non-standard JSON serialization in some environments.
   - **Solution**: Wrapped the expression with `bool(...)` to guarantee a native Python boolean is stored in the output dictionary.

7. **Deferred Gaze Telemetry Integration (Resolved - May 05)**:
   - **Problem**: The specification mentions using raw eye-tracking telemetry (`gaze_x`, `gaze_y` arrays) from Aria glasses datasets as a high-precision fallback for social referencing detection. However, this requires complex `.vrs` parsing and a 200MB dependency, which only benefits a fraction of the dataset.
   - **Solution**: Deferred the implementation of Aria gaze telemetry parsing to a future update to maintain a lighter footprint and rely on optical flow and bounding box centering, which covers all clips uniformly. Added a "Future Possible Direction" section in the Implementation Strategy.

8. **Optical Flow Downsampling Artifacts (Resolved - May 05)**:
   - **Problem**: The Farneback optical flow computation was heavily downsampling frames by a factor of 0.25 to maintain low latency. This aggressive reduction caused subtle micro-pans to fall below the optical flow noise floor, potentially missing important gaze shifts towards bystanders.
   - **Solution**: Updated `_extract_camera_shift` in `pipeline.py` to use a 0.5x downsampling factor. This recovers significant spatial resolution for detecting micro-pans with an acceptable performance trade-off on the Mac mini M4 Pro.

9. **Camera Shift Magnitude Threshold Tuning (Resolved - May 05)**:
   - **Problem**: The `social_reference_sought` logic used an absolute shift magnitude threshold of `30.0` pixels, which was calibrated for 480p clips but was overly permissive and prone to false positives for higher resolution (e.g. 1080p, 4K) footage.
   - **Solution**: Replaced the static absolute threshold in `process_video` with a resolution-normalized threshold dynamically computed as 4% of the frame's diagonal (`threshold = frame_diagonal * 0.04`). This properly scales the sensitivity across varying video resolutions.

## ⚠️ Unresolved Issues & Suggestions

*None at this time.*
