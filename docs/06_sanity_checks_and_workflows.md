# AI Task Breakdown: Real-Time Sanity Checks & Eng Workflow

## Objective
Implement infrastructure safety mechanisms, visual debugging, and metric checks to prevent "Silent Failures" commonly caused by extreme memory pressure on 24GB unified memory environments.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Visual Debugger Overlay (`debug_overlay.py`)
- **Action**: Create `debug_overlay.py` utilizing the `OpenCV` (`cv2`) library.
- **Action**: Write a rendering function `generate_debug_frame(frame, action_labels, pose_landmarks, vlm_rating)`.
- **Action**: Utilize `cv2.line`, `cv2.circle`, and `cv2.putText` to structurally draw the pose skeleton, active Charades action class labels (as text overlay), and VLM cognitive state rating on top of the RGB frame array.
- **Action**: The debug overlay should run on **third-person video frames** (since that's where pose and facial analysis occur). Optionally also render the egocentric view side-by-side for developer reference.
- **Action**: Write a pipeline hook `export_debug_video()` triggered automatically when `processed_clips_count % 100 == 0`. Save to `~/charades_ego_data/debug/`.

### Task 2: Distribution Monitoring (The Flatline Check)
- **Action**: Inside the central runner loop, utilize a rolling buffer queue (e.g., `from collections import deque; score_buffer = deque(maxlen=20)`).
- **Action**: Implement an anomaly detection function `check_score_variance_for_collapse(score_buffer)`. Calculate standard deviation via `numpy.std()`.
- **Action**: If `std < 0.1` and the buffer is full, trigger a `ModelCollapseError` to prevent writing 1000s of biased identical VLM results if the VLM loops.


### Task 3: Confidence Routing (Gemma 4 E2B)
- **Action**: Ensure that Modules 03 (Engagement VLM) and 04 (Flinch Pose) augment their returns with a `Confidence_Score` (0.0 - 1.0).
- **Action**: Use the ultra-fast **Gemma 4 E2B** model to cross-check Module 3's output and detect hallucinations, thereby determining the VLM's true output confidence.
- **Action**: In the main loop, calculate the minimum confidence after Gemma 4 E2B adjustments: `min_confidence = min(adjusted_vlm_conf, pose_conf)`.
- **Action**: Write logic: If `min_confidence < 0.6`, invoke `shutil.copy` to copy the third-person clip to `~/charades_ego_data/manual_review/` and **exclude** the clip from the final parquet export. Do **not** move or delete the original video — it may be needed for re-processing.

### Task 4: The "Reaction Lag" Temporal Filter
- **Action**: Write a function `validate_reaction_lag(action_start_timestamp, flinch_frame_timestamp)`.
- **Action**: Calculate the delay `lag_ms = (flinch_frame_timestamp - action_start_timestamp) * 1000`.
- **Action**: Implement the logic switch:
  - If `lag_ms < 100` milliseconds → discard as noise/coincidence (human bio-reaction minimum is ~150–200ms).
  - If `lag_ms > 5000` milliseconds → discard as unrelated to the action (reaction too delayed to be causal).
- **Success Criteria**: All pipeline safeguards are properly modularized and actively intercept hallucinated, collapsed, or physically impossible system conclusions before they enter the final parquet.
