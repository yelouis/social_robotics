# AI Task Breakdown: The Multi-Modal Perception Module (`calculate_body_kinematics.py`)

## Objective
Implement body pose tracking and kinesthetic "Flinch" detection as Node 3 of the waterfall filter. This module operates on the **third-person paired video** (same as Module 03) using the same `(t_start, t_end)` temporal analysis window.

## ⚠️ Critical: Use the Third-Person Paired Video
Just like the Engagement Module, this module **must not** attempt to track the actor's body from the egocentric clip — their torso, shoulders, and head are not visible in first-person. Load the third-person video via `tp_video_path` from `paired_clips_manifest.csv`.

## Status: ✅ COMPLETED
- **Implemented**: April 18, 2026
- **Primary Backend**: YOLOv8-Pose (`ultralytics`)
- **Validation**: Cross-referenced with Node 1 (Attention) and verified on M4 Pro.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Pose Estimation Integration
- **Action**: Create the file `calculate_body_kinematics.py`. [DONE]
- **Action**: Load the list of clips from `engagement_results.json` (output of Module 03). Each entry provides `tp_video_id`, `tp_video_path` (via manifest), `t_start`, and `t_end`. [DONE]
- **Action**: Integrate a lightweight, low-RAM Pose estimation library. **Pivot**: Used `ultralytics` (YOLOv8-Pose) due to MediaPipe stability issues on this environment. [DONE]
- **Action**: Write a loop that processes **third-person video** frames within the `(t_start, t_end)` window. Save the coordinates of major upper-body landmarks. [DONE]

### Task 2: Kinematic Flinch Calculation
- **Action**: Write a mathematical function `calculate_peak_velocity(pose_history, fps)`. [DONE]
- **Action**: Implement the formula: `V_peak = max(delta_pose / delta_t)`. [DONE]
- **Action**: Expose a configurable `FLINCH_THRESHOLD` variable. Default set to `150.0`. [DONE]

### Task 3: Grounded Attention Check Integration
- **Action**: Write a combined handler function `evaluate_multimodal_state(ego_video_id, pose_data)`. [DONE]
- **Action**: Cross-reference with `attention_results.json`. Only process clips where `passed_attention == true`. [DONE]

### Task 4: Signal Prioritization & Output
- **Action**: Implement the priority logic: If `V_peak > FLINCH_THRESHOLD`, label the clip as `FLINCH_DETECTED = True`. [DONE]
- **Action**: Save the output to `flinch_results.json`. [DONE]
- **Action**: The `confidence` field reflects pose detection quality. [DONE]

## Success Criteria
- ✅ System returns `flinch: true` for sudden Movements.
- ✅ YOLOv8-Pose runs efficiently within the M4 Pro's RAM budget.
