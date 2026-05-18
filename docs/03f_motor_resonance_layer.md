# AI Task Breakdown: Motor Resonance Layer (03f)

## Objective
The **Motor Resonance Layer** captures "Empathy and Mirroring." Infants learn affective states via Mirror Neurons—if a baby drops something and gets scared, a parent visibly winces or flinches in sympathy. This layer compares the chaotic kinematics of the POV camera (representing the actor's trauma or abrupt action) against the reactionary pose kinematics of the bystander.

---

## 📥 Input Requirements
- **`filtered_manifest.json`**: For bystander location arrays.
- **Raw Video Chunk**: Bounded around the specific task window.

---

## 🛠️ Implementation Strategy

### 1. POV Kinematic Extraction (EgoMotion)
First, we must quantify the severity of the POV actor's physical state.
- **Mechanism**: Compute dense Optical Flow (`cv2.calcOpticalFlowFarneback`) on the background pixels across consecutive frames.
- **Metric**: High, chaotic variance in the global optical flow indicates the POV actor tripped, dropped something suddenly, or violently shook the camera (High Ego-Kinetic Energy).

### 2. Bystander Pose Extraction (YOLOv8 Pose)
We must track how the bystander responds physically. 
- **Recommended SOTA Toolkit**: Use the **Ultralytics YOLO** framework, loading the **YOLOv8-pose** model architecture via the tier-per-host registry (`get_model("layer_03f_pose")`, defaulting to `yolov8x-pose.pt`). The `x` tier maximizes keypoint accuracy on partially-occluded bystanders, which is the dominant failure mode for flinch metrics. It replaces earlier MMPose/RTMPose specifications to avoid complex `mmcv` compilation failures on Apple Silicon.
- **Mechanism**: Run YOLOv8-pose on the bystander's bounding box during the reaction window. It natively supports PyTorch MPS tensors on Apple Silicon (validated on the **Mac Studio M4 Max** target host) for target real-time FPS.

### 3. Correlating the Resonance
- **The Flinch Metric**: Calculate the velocity of the bystander's wrist/shoulder keypoints. If they rapidly elevate (throwing hands up defensively) within `0.5s` of a spike in POV Ego-Kinetic Energy, we have detected a sympathetic physical flinch. 
- **Mirroring Metric**: If the EgoMotion pans down rapidly (POV person leaning over), and the bystander's spine keypoints (shoulder to hip) angle inward congruently, they are physically mirroring the intention.

---

## 📤 Output Schema and Integration
**Example Output Data (`03f_motor_resonance_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03f_motor_resonance",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "ego_kinetic_chaos_score": 0.88,
      "per_person": [
        {
          "person_id": 0,
          "bystander_pose_velocity_peak": 4.5,
          "resonance_delay_sec": 0.2,
          "motor_resonance_detected": true,
          "empathy_scalar": 0.92
        }
      ]
    }
  ]
}
```

## Verification & Validation Check
- **Singular Video Test**: Extract a known "trip and fall" clip. Render a debug video that outputs the EgoMotion scalar text on the top-left and draws the YOLOv8 pose skeleton over the bystander. Verify the `bystander_pose_velocity_peak` triggers immediately after the camera jolt.
- **Batch Test**: Run a subset iteration monitoring YOLOv8 inference speed. Ensure the model instantiation uses PyTorch MPS tensors correctly on the **Mac Studio (M4 Max, 64 GB unified memory)** so that operations process at target real-time FPS without falling back to slow CPU loops.

---

## 🧪 Resolved Issues & Implementation Refinements

1. **Keypoint Index Misalignment Across Frames (Resolved - May 04)**:
   - **Problem**: Bystander pose velocity was computed by comparing keypoints stored in a flat list ordered by insertion. When different keypoints passed the confidence threshold across consecutive frames (e.g., frame N has L-shoulder + R-wrist, frame N+1 has R-shoulder + R-wrist), the list index `0` mapped to different body parts. This produced meaningless cross-body velocity values (e.g., L-shoulder vs. R-shoulder distance), inflating or deflating the `bystander_pose_velocity_peak` metric arbitrarily.
   - **Solution**: Refactored keypoint storage from a flat list to a `dict` keyed by the COCO keypoint index (`{5: (x,y), 6: (x,y), ...}`). Velocity is now computed only over the intersection of keypoint indices present in both consecutive frames, guaranteeing body-part-to-body-part correspondence.

2. **False-Positive EgoMotion Spikes on Calm Video (Resolved - May 04)**:
   - **Problem**: The spike detection used a relative threshold (`> 70% of max`), which meant even a completely still video with micro-noise would always produce "spikes" near its trivial maximum. This led to downstream false-positive `motor_resonance_detected: true` results on calm, non-eventful footage.
   - **Solution**: Added an absolute chaos floor threshold (`max_chaos_score < 3.0`). If the maximum optical flow magnitude across the entire reaction window stays below 3.0 pixels/frame, the method returns an empty spike list. The relative 70% threshold only applies after passing this floor, ensuring only genuinely chaotic camera movements trigger downstream resonance correlation.

3. **Crop-Local Pixel Velocity Without Size Normalization (Resolved - May 04)**:
   - **Problem**: Keypoint coordinates from YOLOv8-pose are in crop-local pixel space. Since bystander bounding boxes vary in size across frames (e.g., 50×50 vs. 300×400), the same physical arm movement would produce drastically different pixel distances depending on the crop resolution. The velocity normalization divisor (`/ 100.0`) was an arbitrary pixel constant that did not account for crop scale, making `bystander_pose_velocity_peak` unreliable across different bystander distances and zoom levels.
   - **Solution**: Each keypoint coordinate is now divided by the crop's diagonal length (`sqrt(w² + h²)`) at inference time, converting positions to a scale-invariant `[0, 1]` space. The velocity normalization divisor was updated from `/ 100.0` (pixels) to `/ 0.5` (diagonals/sec), where 0.5 diagonals/sec represents a significant flinch.

4. **Heuristic Constants Hoisted to Class-Level Tuning Block (Resolved - May 08)**:
   - **Problem**: Magic numbers governing classification decisions were inlined as literals throughout the detection path: chaos floor `3.0`, max-chaos normalizer `20.0`, relative spike threshold `0.7`, frame-resize factor `0.5`, keypoint confidence threshold `0.5`, velocity normalizer `0.5`, velocity cap `10.0`, reaction-window cap `0.5s`, empathy normalizer `5.0`, mirroring vertical-flow threshold `-1.0`, mirroring time window `±0.5s`, mirroring-delta threshold `0.1`, and mirroring-scalar normalizer `0.5`. Tuning the pipeline required source edits at multiple sites.
   - **Solution**: Hoisted all heuristic constants into a `# --- Detection Tuning ---` block at the top of `MotorResonancePipeline`. Subclass-based ablations and per-experiment overrides become a matter of attribute assignment rather than source edits; zero runtime overhead, no schema change.

5. **Skipped Videos Reprocessed on Every Resume (Resolved - May 08)**:
   - **Problem**: When `process_video` returned `None` (file missing, no bystanders, etc.), the video was skipped and not marked as processed, causing resume runs to re-execute substantial MPS time on guaranteed no-op work.
   - **Solution**: Introduced a `_sentinel(video_id, reason)` helper that returns explicit skipped reasons. Every video — including skipped ones — is written atomically and added to `self.processed_ids`. Resume cost for previously-skipped videos drops to set-membership lookup, and downstream consumers gain explicit visibility into *why* a video was excluded.

6. **Adaptive Frame Stride for Optical Flow (Resolved - May 08)**:
   - **Problem**: Computing dense Farneback flow on *every* frame in the reaction window dominated wall-clock cost and scaled poorly. 
   - **Solution**: Added a `TARGET_FLOW_FPS = 10.0` class constant. `_extract_ego_motion` computes `frame_stride = max(1, round(fps / TARGET_FLOW_FPS))` and seeks/reads frames in stride-sized hops. At 30 FPS this is a 3× speedup with negligible loss in jolt detectability.

7. **`mean_v` Pollution by Foreground Motion in Mirroring Signal (Resolved - May 08)**:
   - **Problem**: The vertical optical-flow signal was computed over the *entire* downsampled frame, including the bystander's body. The "Mirroring Metric" used this signal to gate detection. When the bystander was large in frame, their own vertical motion contaminated the global mean.
   - **Solution**: Added a `_bystander_mask_for_frame` helper that builds a boolean mask zeroing out the nearest-timestamp bbox for every bystander. `_extract_ego_motion` computes `mean_v` over background pixels only.

8. **YOLOv8 First-Detection Selection Without Bystander Filtering (Resolved - May 08)**:
   - **Problem**: The pipeline unconditionally selected the first person detection in the cropped bystander bbox. When the crop contained multiple persons, this could track the wrong person across frames.
   - **Solution**: Added a `_select_pose_detection(results, crop_w, crop_h)` helper that computes IoU between each detection's bbox and the original input crop reference, picking the detection with the highest IoU as the intended bystander.

9. **Mirroring Detection Inherits the Chaos-Floor False-Negative (Resolved - May 08)**:
   - **Problem**: The chaos floor (`max_chaos_score < 3.0`) returned an empty spike list, suppressing every ego spike. Because mirroring was gated on ego spikes, a calm downward camera tilt (legitimate mirroring) produced no chaos spike and therefore no mirroring detection.
   - **Solution**: `_extract_ego_motion` returns three parallel signals: `chaos_spikes`, `vertical_flow_timeline`, and `norm_max_chaos`. `_correlate_mirroring` consumes `vertical_flow_timeline` directly, independent of the chaos floor.

10. **`prev_kpts_by_idx` Reset Broke Velocity Chain on Sparse Detection Failures (Resolved - May 08)**:
    - **Problem**: The pose extraction loop assigned `prev_kpts_by_idx = current_kpts_by_idx` unconditionally, resetting the velocity chain when frames dropped. The peak velocity from the actual flinch could be missed entirely.
    - **Solution**: `prev_kpts_by_idx` and `prev_t` are only updated when `current_kpts_by_idx` is non-empty. To prevent stale comparisons across long occlusions, the velocity computation is gated on `dt <= PREV_KPTS_CARRY_FORWARD_SEC` (0.5 s).

