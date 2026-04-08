# AI Task Breakdown: The Multi-Modal Perception Module (Gaze + Body)

## Objective
Implement body pose tracking and kinesthetic "Flinch" detection formulas. This acts as the second, medium-compute stage in the serialized batch process.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Pose Estimation Integration
- **Action**: Create the file `calculate_body_kinematics.py`.
- **Action**: Integrate a lightweight, low-RAM Pose estimation library (e.g., MediaPipe Pose or YOLOv8-Pose).
- **Action**: Write a loop that processes valid video frames and saves the coordinates of major upper-body landmarks (shoulders, head, torso).

### Task 2: Kinematic Flinch Calculation
- **Action**: Write a mathematical function `calculate_peak_velocity(pose_data_over_time)`.
- **Action**: Implement the formula: `V_peak = max(delta_pose / delta_t)`. Calculate `delta_pose` as the maximum euclidean distance shift of the upper body keypoints between `frame(t)` and `frame(t-1)`.
- **Action**: Expose a configurable `FLINCH_THRESHOLD` variable at the top of the file.

### Task 3: Grounded Attention Check Integration
- **Action**: Write a combined handler function `evaluate_multimodal_state(clip_id, pose_data)`.
- **Action**: Pull the `ATTENTION_FALSE/TRUE` result previously calculated by Module 2. If it is `ATTENTION_FALSE`, discard the calculated physical emotion data (label the reaction as ungrounded 'Noise').

### Task 4: Signal Prioritization Logic
- **Action**: Implement the priority logic: If `V_peak > FLINCH_THRESHOLD`, label the clip as `SOCIAL_VIOLATION: FLINCH_DETECTED = True`. 
- **Action**: Return this data as an object `{clip_id: str, flinch: bool, v_peak: float}`.
- **Success Criteria**: The system returns `Flinch: True` for fast, anomalous movements and `Flinch: False` for smooth/still frames, filtering successfully on lower RAM constraints before heavy VLM tasks.
