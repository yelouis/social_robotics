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
- **Mechanism**: Compute dense Optical Flow (`cv2.calcOpticalFlowFarneback`) across the reaction window, temporally subsampled to `TARGET_FLOW_FPS = 10` (a ~3× cost cut at 30 FPS source with negligible loss in jolt detectability — impulsive jolts survive stride sampling).
- **Metric**: High, chaotic variance in the optical flow (95th-percentile magnitude) indicates the POV actor tripped, dropped something suddenly, or violently shook the camera (High Ego-Kinetic Energy).
- **Absolute Chaos Floor**: Spike detection is *relative* (`> 70 % of the window max`), so a completely still video's micro-noise would always produce "spikes" near its trivial maximum. An absolute floor (`CHAOS_FLOOR = 3.0` px/frame on the window max) is applied **before** the relative threshold, so only genuinely chaotic camera motion can trigger downstream resonance correlation. Do not remove the floor — it is what prevents false-positive `motor_resonance_detected` on calm footage.
- **Bystander Masking (vertical flow)**: The mean vertical-flow signal that the Mirroring Metric consumes is computed over *background pixels only* — `_bystander_mask_for_frame` zeroes out the nearest-timestamp bbox of every bystander, so a large in-frame bystander's own motion cannot masquerade as a camera tilt. (The chaos percentile itself is currently *not* masked — see Unresolved Issue 2.)

### 2. Bystander Pose Extraction (YOLOv8 Pose)
We must track how the bystander responds physically. 
- **Recommended SOTA Toolkit**: Use the **Ultralytics YOLO** framework, loading the **YOLOv8-pose** model architecture via the tier-per-host registry (`get_model("layer_03f_pose")`, defaulting to `yolov8x-pose.pt`). The `x` tier maximizes keypoint accuracy on partially-occluded bystanders, which is the dominant failure mode for flinch metrics. It replaces earlier MMPose/RTMPose specifications to avoid complex `mmcv` compilation failures on Apple Silicon.
- **Mechanism**: Run YOLOv8-pose on the bystander's bounding box during the reaction window. It natively supports PyTorch MPS tensors on Apple Silicon (validated on the **Mac Studio M4 Max** target host) for target real-time FPS.

### 3. Correlating the Resonance
- **The Flinch Metric**: Calculate the velocity of the bystander's wrist/shoulder keypoints. If they rapidly elevate (throwing hands up defensively) within `0.5s` of a spike in POV Ego-Kinetic Energy, we have detected a sympathetic physical flinch.
- **Scale-Invariant Velocity**: Keypoint coordinates are divided by the crop's diagonal length at inference time, converting positions into a `[0, 1]` space before differencing. This is load-bearing: raw crop-local pixels made the same physical arm movement produce drastically different "velocities" depending on bystander distance/crop size; the normalizer (`VELOCITY_NORMALIZER = 0.5` diagonals/sec ≈ a significant flinch) is calibrated for the normalized space and must not be re-pointed at raw pixels.
- **Mirroring Metric**: If the EgoMotion pans down rapidly (POV person leaning over), and the bystander's spine keypoints (shoulder to hip) angle inward congruently, they are physically mirroring the intention.
- **Tuning Surface**: All heuristic thresholds live in the class-level `# --- Detection Tuning ---` block of `MotorResonancePipeline`, so per-experiment ablations are subclass attribute overrides rather than source edits.

---

## 📤 Output Schema and Integration
Videos that legitimately produce no output (missing file, no bystanders/tasks, no ego spikes, no pose data) emit **sentinel records** (`"tasks_analyzed": [], "skipped_reason": "..."`) and are marked processed, so resume runs skip them instead of re-paying MPS time, and downstream consumers can see *why* a video was excluded.

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

*(June 12 cleanup: five earlier entries — the absolute chaos floor, crop-diagonal velocity normalization, the class-level tuning block, sentinel records, and background-masked vertical flow — were design changes and have been integrated into the Implementation Strategy / Output Schema sections above with their rationale. The entries below are retained because they document subtle failure modes likely to recur if the mechanisms are refactored.)*

1. **Keypoint Index Misalignment Across Frames (Resolved - May 04)**:
   - **Problem**: Bystander pose velocity was computed by comparing keypoints stored in a flat list ordered by insertion. When different keypoints passed the confidence threshold across consecutive frames (e.g., frame N has L-shoulder + R-wrist, frame N+1 has R-shoulder + R-wrist), the list index `0` mapped to different body parts. This produced meaningless cross-body velocity values (e.g., L-shoulder vs. R-shoulder distance), inflating or deflating the `bystander_pose_velocity_peak` metric arbitrarily.
   - **Solution**: Refactored keypoint storage from a flat list to a `dict` keyed by the COCO keypoint index (`{5: (x,y), 6: (x,y), ...}`). Velocity is now computed only over the intersection of keypoint indices present in both consecutive frames, guaranteeing body-part-to-body-part correspondence.

2. **Adaptive Frame Stride for Optical Flow (Resolved - May 08)**:
   - **Problem**: Computing dense Farneback flow on *every* frame in the reaction window dominated wall-clock cost and scaled poorly.
   - **Solution**: Added a `TARGET_FLOW_FPS = 10.0` class constant. `_extract_ego_motion` computes `frame_stride = max(1, round(fps / TARGET_FLOW_FPS))` and advances in stride-sized hops. At 30 FPS this is a 3× speedup with negligible loss in jolt detectability. *(Note: the stride currently advances via per-step `cap.set` seeking — see Unresolved Issue 1.)*

3. **YOLOv8 First-Detection Selection Without Bystander Filtering (Resolved - May 08)**:
   - **Problem**: The pipeline unconditionally selected the first person detection in the cropped bystander bbox. When the crop contained multiple persons, this could track the wrong person across frames.
   - **Solution**: Added a `_select_pose_detection(results, crop_w, crop_h)` helper that computes IoU between each detection's bbox and the original input crop reference, picking the detection with the highest IoU as the intended bystander.

4. **Mirroring Detection Inherits the Chaos-Floor False-Negative (Resolved - May 08)**:
   - **Problem**: The chaos floor returned an empty spike list, suppressing every ego spike. Because mirroring was gated on ego spikes, a calm downward camera tilt (legitimate mirroring) produced no chaos spike and therefore no mirroring detection.
   - **Solution**: `_extract_ego_motion` returns three parallel signals: `chaos_spikes`, `vertical_flow_timeline`, and `norm_max_chaos`. `_correlate_mirroring` consumes `vertical_flow_timeline` directly, independent of the chaos floor.

5. **`prev_kpts_by_idx` Reset Broke Velocity Chain on Sparse Detection Failures (Resolved - May 08)**:
   - **Problem**: The pose extraction loop assigned `prev_kpts_by_idx = current_kpts_by_idx` unconditionally, resetting the velocity chain when frames dropped. The peak velocity from the actual flinch could be missed entirely.
   - **Solution**: `prev_kpts_by_idx` and `prev_t` are only updated when `current_kpts_by_idx` is non-empty. To prevent stale comparisons across long occlusions, the velocity computation is gated on `dt <= PREV_KPTS_CARRY_FORWARD_SEC` (0.5 s).

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Per-stride `cap.set` seeking in `_extract_ego_motion` (H.264 keyframe-approximate anti-pattern)
**Status**: ⚠️ Confirmed Unresolved — Found in the June 12 repo audit. `_extract_ego_motion` (`src/layer_03f_motor_resonance/pipeline.py`, the `if frame_stride > 1: cap.set(CAP_PROP_POS_FRAMES, current_frame_idx)` inside the stride loop) seeks on **every stride step** (~10 Hz across the whole reaction window). This is the exact pattern the project eliminated everywhere else: on H.264, `cap.set` is keyframe-approximate — each seek re-decodes from the nearest keyframe (~10–20× wasted decode) **and** can desync frame↔timestamp, which corrupted 03a scores before its no-seek rewrite (03a Resolved #5; same fix validated bit-identical in 02 Resolved #18). At stride 3, sequentially `grab()`-ing 2 frames and `read()`-ing the third is strictly cheaper and frame-accurate. The sparse per-detection seeks in `_extract_and_correlate_pose` are fine (the face-quality-prefilter precedent: widely-spaced samples are the right use of seeking).

**Remediation (single option — proven fix)**: Replace the per-step seek with the sequential `grab()`/`read()` skip pattern used by `shared/climax_extraction.py` and 03a, and validate on a few cached clips that chaos spike timestamps and `ego_kinetic_chaos_score` are unchanged (the established bit-identical protocol).
  - *Pros*: Removes wasted keyframe re-decode and the timestamp-desync risk; pattern already validated twice in this repo; no new dependency.
  - *Cons*: None of substance — mechanical change plus a short validation run.

Your selection: _____

---

### Issue 2: Ego chaos score is not bystander-masked — a flinching bystander can correlate with itself
**Status**: ⚠️ Confirmed Unresolved — Found in the June 12 repo audit. The chaos metric (`chaos_score = np.percentile(mag, 95)` in `_extract_ego_motion`) is computed over the **full downsampled frame**, while only the vertical-flow (mirroring) signal applies `_bystander_mask_for_frame`. Two consequences: (a) the Implementation Strategy's claim that ego-kinetic energy reflects the *POV actor's* motion is not strictly true — a large in-frame bystander's abrupt movement raises the "ego" chaos; and (b) a **self-correlation false positive** is possible: the bystander's own flinch creates the chaos spike, and `_extract_and_correlate_pose` then detects that same flinch within `RESONANCE_WINDOW_SEC` of "the wearer's" spike, yielding `motor_resonance_detected: true` with no actual wearer event. The flinch metric's core premise (bystander reacts *to the wearer*) is inverted in that case.

**Option A (recommended)**: **Mask the chaos percentile like the vertical flow.** Compute `np.percentile(mag[mask], 95)` with the existing `_bystander_mask_for_frame` (falling back to the full frame when the mask is empty, as `mean_v` already does).
  - *Pros*: One-line change reusing existing machinery; makes the metric match its documented meaning; eliminates the self-correlation mechanism.
  - *Cons*: When a bystander dominates the frame, few background pixels remain and the percentile gets noisier (mitigated by the existing `mask.any()` fallback); chaos floor/normalizer (3.0/20.0) may need a small re-calibration since masked percentiles run slightly lower.

**Option B**: **Keep full-frame chaos but require corroboration.** Keep the metric as-is and only accept a spike when the masked background flow also spikes at the same timestamp.
  - *Pros*: No re-calibration of the chaos constants; explicitly separates "scene got chaotic" from "camera moved".
  - *Cons*: Two thresholds instead of one; more logic for the same outcome Option A achieves directly.

Your selection: _____

