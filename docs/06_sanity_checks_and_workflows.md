# AI Task Breakdown: Real-Time Sanity Checks & Eng Workflow

## Objective
Implement infrastructure safety mechanisms, visual debugging, and metric checks to prevent "Silent Failures" commonly caused by extreme memory pressure on 16GB unified memory environments.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Visual Debugger Overlay (`debug_overlay.py`)
- **Action**: Create `debug_overlay.py` utilizing the `OpenCV` (`cv2`) library.
- **Action**: Write a rendering function `generate_debug_frame(frame, gaze_vector, pose_landmarks, vlm_rating)`.
- **Action**: Utilize `cv2.line`, `cv2.circle`, and `cv2.putText` to structurally draw the human's gaze vector, pose skeleton, and VLM text on top of the RGB frame array.
- **Action**: Write a pipeline hook `export_debug_video()` triggered automatically when `processed_clips_count % 100 == 0`.

### Task 2: Distribution Monitoring (The Flatline Check)
- **Action**: Inside the central runner loop, utilize a rolling buffer queue (e.g., `from collections import deque; score_buffer = deque(maxlen=20)`).
- **Action**: Implement an anomaly detection function `check_score_variance_for_collapse(score_buffer)`. Calculate standard deviation numpy `.std()`.
- **Action**: If `std < 0.1` and the buffer is full, trigger a `ModelCollapseError` to prevent writing 1000s of biased identical VLM results if the VLM loops.

### Task 3: Confidence Routing
- **Action**: Ensure that Modules 2 (CV), 3 (VLM), and 4 (Pose) augment their returns with a `Confidence_Score` (0.0 - 1.0).
- **Action**: In the main loop, calculate the average or minimum confidence.
- **Action**: Write logic: If `Total_Confidence < 0.6`, invoke `os.rename` or `shutil.move` to place the raw clip video inside `~/ego4d_data/manual_review/` and exclude it from the parquet export.

### Task 4: The "Reaction Lag" Temporal Filter
- **Action**: Write a function `validate_reaction_lag(action_start_timestamp, flinch_timestamp)`.
- **Action**: Calculate the delay `lag_ms = flinch_timestamp - action_start_timestamp`.
- **Action**: Implement the logic switch: If `lag_ms < 100` milliseconds, discard the reaction as disconnected hallucination / noise, since human bio-reaction minimum boundaries are ~200ms.
- **Success Criteria**: All pipeline safeguards are properly modularized and actively intercept hallucinated, collapsed, or physically impossible system conclusions rapidly.
