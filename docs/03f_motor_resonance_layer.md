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
- **Recommended SOTA Toolkit**: Use the **Ultralytics YOLO** framework, specifically loading the **YOLOv8-pose** model architectures (`yolov8n-pose.pt`). 
- **Mechanism**: Run YOLOv8-pose on the bystander's bounding box during the reaction window. It natively supports PyTorch MPS tensors on the M4 Pro for target real-time FPS.

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
- **Batch Test**: Run a subset iteration monitoring YOLOv8 inference speed. Ensure the model instantiation uses PyTorch MPS tensors correctly on the **24GB RAM Mac mini M4 Pro** so that operations process at target real-time FPS without falling back to slow CPU loops.

---

## 🧪 Resolved Issues & Implementation Refinements

1. **Pose Model Dependency Substitution (Resolved - May 04)**:
   - **Problem**: The original specification called for `MMPose` and `RTMPose` models, but `mmpose` and its required compilation component `mmcv` were not installed in the local environment and are prone to complex build failures on Apple Silicon (M4 Pro).
   - **Solution**: Substituted `MMPose` with `ultralytics` YOLOv8-pose (`yolov8n-pose.pt`), which natively utilizes the PyTorch MPS backend for high-performance inference, avoiding complex compilation errors and fulfilling the real-time processing constraints.

2. **Keypoint Index Misalignment Across Frames (Resolved - May 04)**:
   - **Problem**: Bystander pose velocity was computed by comparing keypoints stored in a flat list ordered by insertion. When different keypoints passed the confidence threshold across consecutive frames (e.g., frame N has L-shoulder + R-wrist, frame N+1 has R-shoulder + R-wrist), the list index `0` mapped to different body parts. This produced meaningless cross-body velocity values (e.g., L-shoulder vs. R-shoulder distance), inflating or deflating the `bystander_pose_velocity_peak` metric arbitrarily.
   - **Solution**: Refactored keypoint storage from a flat list to a `dict` keyed by the COCO keypoint index (`{5: (x,y), 6: (x,y), ...}`). Velocity is now computed only over the intersection of keypoint indices present in both consecutive frames, guaranteeing body-part-to-body-part correspondence.

3. **False-Positive EgoMotion Spikes on Calm Video (Resolved - May 04)**:
   - **Problem**: The spike detection used a relative threshold (`> 70% of max`), which meant even a completely still video with micro-noise would always produce "spikes" near its trivial maximum. This led to downstream false-positive `motor_resonance_detected: true` results on calm, non-eventful footage.
   - **Solution**: Added an absolute chaos floor threshold (`max_chaos_score < 3.0`). If the maximum optical flow magnitude across the entire reaction window stays below 3.0 pixels/frame, the method returns an empty spike list. The relative 70% threshold only applies after passing this floor, ensuring only genuinely chaotic camera movements trigger downstream resonance correlation.

4. **Crop-Local Pixel Velocity Without Size Normalization (Resolved - May 04)**:
   - **Problem**: Keypoint coordinates from YOLOv8-pose are in crop-local pixel space. Since bystander bounding boxes vary in size across frames (e.g., 50×50 vs. 300×400), the same physical arm movement would produce drastically different pixel distances depending on the crop resolution. The velocity normalization divisor (`/ 100.0`) was an arbitrary pixel constant that did not account for crop scale, making `bystander_pose_velocity_peak` unreliable across different bystander distances and zoom levels.
   - **Solution**: Each keypoint coordinate is now divided by the crop's diagonal length (`sqrt(w² + h²)`) at inference time, converting positions to a scale-invariant `[0, 1]` space. The velocity normalization divisor was updated from `/ 100.0` (pixels) to `/ 0.5` (diagonals/sec), where 0.5 diagonals/sec represents a significant flinch.

5. **Missing VideoCapture Guard in Pose Extraction (Resolved - May 04)**:
   - **Problem**: `_extract_and_correlate_pose` did not validate `cap.isOpened()` or `fps == 0` before proceeding with frame reads. If a video file was corrupted or the codec was unsupported, `fps` would be `0`, causing `frame_idx = int(t * 0)` to always seek to frame 0 and silently produce incorrect results rather than failing fast.
   - **Solution**: Added explicit `cap.isOpened()` and `fps == 0` checks at the top of the method, returning `None` early with proper `cap.release()` cleanup.

## ⚠️ Unresolved Issues & Suggestions

- **RTMPose / MMPose Native Integration**: Compiling `mmcv` from source to support `MMPose` on Apple Silicon (M4 Pro) frequently results in obscure C++ errors. If RTMPose's SimCC architecture is definitively required over YOLOv8-pose for superior shoulder/wrist precision, we suggest investigating pre-compiled Conda packages (`conda install mmcv -c open-mmlab`) or community-maintained Apple Silicon wheels, as cloud GPU fallback is not acceptable for this project's local-first architecture.
- **Mirroring Metric Not Implemented**: The documentation specifies a "Mirroring Metric" that detects congruent spine-angle changes (shoulder-to-hip) when the POV person leans over. The current implementation only detects the "Flinch Metric" (rapid wrist/shoulder velocity). Implementing the mirroring metric requires tracking hip keypoints (COCO indices 11, 12) and computing spine angle deltas correlated with downward EgoMotion direction, which is architecturally feasible but was deferred to a future iteration.

