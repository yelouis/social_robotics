# AI Task Breakdown: The Multi-Modal Perception Module (`calculate_body_kinematics.py`)

## Objective
Implement body pose tracking and kinesthetic "Flinch" detection as Node 3 of the waterfall filter. This module operates on the **third-person paired video** (same as Module 03) using the same `(t_start, t_end)` temporal analysis window.

## ⚠️ Critical: Use the Third-Person Paired Video
Just like the Engagement Module, this module **must not** attempt to track the actor's body from the egocentric clip — their torso, shoulders, and head are not visible in first-person. Load the third-person video via `tp_video_path` from `paired_clips_manifest.csv`.

## Agent Instructions: Step-by-Step Tasks

### Task 1: Pose Estimation Integration
- **Action**: Create the file `calculate_body_kinematics.py`.
- **Action**: Load the list of clips from `engagement_results.json` (output of Module 03). Each entry provides `tp_video_id`, `tp_video_path` (via manifest), `t_start`, and `t_end`.
- **Action**: Integrate a lightweight, low-RAM Pose estimation library (e.g., MediaPipe Pose or YOLOv8-Pose).
- **Action**: Write a loop that processes **third-person video** frames within the `(t_start, t_end)` window. Save the coordinates of major upper-body landmarks (shoulders, head, wrists, torso). Use `cv2.VideoCapture` and seek to `t_start` using `cap.set(cv2.CAP_PROP_POS_MSEC, t_start * 1000)` to avoid decoding the full file.

### Task 2: Kinematic Flinch Calculation
- **Action**: Write a mathematical function `calculate_peak_velocity(pose_data_over_time, fps)`.
- **Action**: Implement the formula: `V_peak = max(delta_pose / delta_t)`. Calculate `delta_pose` as the maximum euclidean distance shift of the upper body keypoints between `frame(t)` and `frame(t-1)`. Derive `delta_t` from the video's actual FPS (`1.0 / fps`).
- **Action**: Expose a configurable `FLINCH_THRESHOLD` variable at the top of the file with a sensible default (e.g., `150.0` pixels/second — to be calibrated empirically).

### Task 3: Grounded Attention Check Integration
- **Action**: Write a combined handler function `evaluate_multimodal_state(ego_video_id, pose_data)`.
- **Action**: Cross-reference with `attention_results.json`. If `passed_attention == false`, discard the calculated physical data (label the reaction as ungrounded 'Noise'). This should be a defensive check — in normal flow, only passing clips reach this module.

### Task 4: Signal Prioritization & Output
- **Action**: Implement the priority logic: If `V_peak > FLINCH_THRESHOLD`, label the clip as `FLINCH_DETECTED = True`. 
- **Action**: Save the output to `flinch_results.json` as a JSON array:
  ```json
  [
    {
      "ego_video_id": "ABCDEEGO",
      "tp_video_id": "ABCDE",
      "flinch": false,
      "v_peak": 42.3,
      "confidence": 0.92
    }
  ]
  ```
- **Action**: The `confidence` field should reflect pose detection quality — average of MediaPipe/YOLO per-landmark confidence scores across analyzed frames.
- **Success Criteria**: The system returns `flinch: true` for sudden, anomalous upper-body movements and `flinch: false` for smooth/still frames. Pose estimation runs within the M1 Pro's RAM budget since MediaPipe/YOLOv8-Pose are far lighter than VLMs.
