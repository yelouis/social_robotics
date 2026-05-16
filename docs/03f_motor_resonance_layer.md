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
- **Recommended SOTA Toolkit**: Use the **Ultralytics YOLO** framework, specifically loading the **YOLOv8-pose** model architectures (`yolov8x-pose.pt`). 
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

6. **RTMPose / MMPose Native Integration (Resolved - May 05)**:
   - **Problem**: The pipeline originally specified RTMPose and MMPose, but `mmcv` compilation fails on Apple Silicon (M4 Pro). The documentation (`ml_dependencies.md`) still listed RTMPose as a requirement, creating a discrepancy between the codebase and specifications.
   - **Solution**: Formally updated `ml_dependencies.md` to designate `ultralytics` YOLOv8-pose (`yolov8n-pose.pt`) as the primary production model and marked RTMPose as Deferred/Optional. This eliminates misleading dependency documentation and standardizes around the functional PyTorch MPS backend.

7. **Mirroring Metric Not Implemented (Resolved - May 05)**:
   - **Problem**: The original specification called for tracking congruent spine-angle changes between the POV camera and the bystander. The implementation lacked the ability to extract hip keypoints, compute spine angles, and correlate them with vertical EgoMotion flow.
   - **Solution**: Expanded keypoint extraction in `pipeline.py` to include COCO hip indices (11/12). Added a `_correlate_mirroring` method that calculates the spine angle (`atan2`) and correlates significant inward angle changes (>0.1) with downward EgoMotion spikes (`v_flow < -1.0`) within the reaction window. Output updated to include `mirroring_detected` and `mirroring_scalar`.

8. **Heuristic Constants Hoisted to Class-Level Tuning Block (Resolved - May 08)**:
   - **Problem**: Magic numbers governing classification decisions were inlined as literals throughout the detection path: chaos floor `3.0`, max-chaos normalizer `20.0`, relative spike threshold `0.7`, frame-resize factor `0.5`, keypoint confidence threshold `0.5`, velocity normalizer `0.5`, velocity cap `10.0`, reaction-window cap `0.5s`, empathy normalizer `5.0`, mirroring vertical-flow threshold `-1.0`, mirroring time window `±0.5s`, mirroring-delta threshold `0.1`, and mirroring-scalar normalizer `0.5`. Tuning the pipeline (e.g., adjusting flinch sensitivity for a different age cohort, recalibrating after a YOLOv8 model swap, or adapting the chaos floor for handheld vs. head-mounted POV) required source edits at multiple sites and risked drift between the detection logic and downstream documentation.
   - **Solution**: Hoisted all heuristic constants into a `# --- Detection Tuning ---` block at the top of `MotorResonancePipeline`: `CHAOS_FLOOR`, `CHAOS_NORMALIZER`, `SPIKE_RELATIVE_THRESHOLD`, `FRAME_RESIZE_FACTOR`, `TARGET_FLOW_FPS`, `KPT_CONFIDENCE_THRESHOLD`, `VELOCITY_NORMALIZER`, `VELOCITY_CAP`, `RESONANCE_WINDOW_SEC`, `EMPATHY_NORMALIZER`, `MIRRORING_VFLOW_THRESHOLD`, `MIRRORING_TIME_WINDOW_SEC`, `MIRRORING_DELTA_THRESHOLD`, `MIRRORING_SCALAR_NORMALIZER`, `PREV_KPTS_CARRY_FORWARD_SEC`, and `RELEVANT_KPT_INDICES`. All call sites in `_extract_ego_motion`, `_extract_and_correlate_pose`, `_correlate_mirroring`, and `_select_pose_detection` now reference the class attributes. Subclass-based ablations and per-experiment overrides become a matter of attribute assignment rather than source edits; zero runtime overhead, no schema change.

9. **Skipped Videos Reprocessed on Every Resume (Resolved - May 08)**:
   - **Problem**: When `process_video` returned `None` (file missing, no bystanders/tasks, no qualifying ego spikes, or every reaction window filtered out), the `if result:` guard in `run` skipped both `results.append(...)` and `self.processed_ids.add(video_id)`. Subsequent resume runs re-iterated these videos, re-executed Farneback optical flow over the entire reaction window, and re-ran YOLOv8-pose inference per sampled frame, only to discard the result again. For batches with many edge-case videos (bystanders out-of-frame, calm videos that fail the chaos floor), this consumed substantial MPS time on guaranteed no-op work.
   - **Solution**: Introduced a `_sentinel(video_id, reason)` helper that returns `{"video_id": ..., "layer": "03f_motor_resonance", "tasks_analyzed": [], "skipped_reason": "file_not_found" | "no_bystanders_or_tasks" | "no_ego_spikes" | "no_pose_data"}`, and replaced the four `return None` sites in `process_video` with calls to this helper. The truthiness gate in `run` was removed so every video — including skipped ones — is appended to `results`, written atomically, and added to `self.processed_ids`. Resume cost for previously-skipped videos drops to set-membership lookup, and downstream consumers gain explicit visibility into *why* a video was excluded. The exception path in `run` retains its existing behavior (errors are logged but not marked processed) so transient failures (e.g., MPS memory pressure) remain retryable on rerun. Consumers must filter on `tasks_analyzed` length or the presence of `skipped_reason` to avoid treating sentinel entries as detection results.

10. **Adaptive Frame Stride for Optical Flow (Resolved - May 08)**:
    - **Problem**: `_extract_ego_motion` seeked to `start_frame = int(start_sec * fps)` and incremented `current_frame_idx` by 1 each iteration, computing dense Farneback flow on *every* frame in the reaction window. For a 30 FPS source and a 5-second window this was 150 dense optical-flow computations per task per video, dominating wall-clock cost and scaling poorly. The pipeline already downsampled each frame to 0.5× resolution but did not subsample temporally.
    - **Solution**: Added a `TARGET_FLOW_FPS = 10.0` class constant; `_extract_ego_motion` now computes `frame_stride = max(1, round(fps / TARGET_FLOW_FPS))` and seeks/reads frames in stride-sized hops. At 30 FPS this is a 3× speedup with negligible loss in jolt detectability (jolts last 100–300 ms and are easily caught by 10 Hz sampling). Calibration handling: `chaos_score` is intentionally left in raw flow-magnitude units regardless of stride — impulsive jolts (frame-pair pixel discontinuities) do not scale with stride the way continuous motion does, and stride-normalizing chaos would have erased the jolt fixture in `tests/test_layer_03f.py::test_ego_motion_detects_jolt`. `mean_v`, by contrast, *does* scale linearly with stride for continuous camera tilt, so it is divided by `frame_stride` to keep `MIRRORING_VFLOW_THRESHOLD` calibrated against per-frame-equivalent rates. This deviates from Option A's literal "multiply by frame_stride" instruction (which is dimensionally backwards for continuous motion) and is documented here so a future audit doesn't reintroduce the error.

11. **`mean_v` Pollution by Foreground Motion in Mirroring Signal (Resolved - May 08)**:
    - **Problem**: The vertical optical-flow signal was computed as `np.mean(flow[..., 1])` over the *entire* downsampled frame, including the bystander's body. The "Mirroring Metric" then used this signal to gate detection of "downward EgoMotion (camera pans down)". When the bystander was large in frame (close-up, multiple bystanders, or a leaning person filling 30 %+ of pixels), their own vertical motion contaminated the global mean, either fabricating a fake "camera pan down" signal or canceling a real one. The documented intent — "Compute dense Optical Flow on the *background* pixels" — was not realized.
    - **Solution**: Added a `_bystander_mask_for_frame(t, frame_shape_ds, bystanders)` helper that builds a boolean mask over the downsampled frame zeroing out the nearest-timestamp bbox for every bystander (rectangles scaled by `FRAME_RESIZE_FACTOR`). `_extract_ego_motion` now accepts the `bystanders` list (plumbed in from `process_video`) and computes `mean_v = np.mean(flow[..., 1][mask])` over background pixels only, falling back to the unmasked mean only when every pixel is inside a bbox (degenerate case). The chaos magnitude `np.percentile(mag, 95)` deliberately remains computed over the full frame — chaos is intended to capture overall scene dynamics for flinch correlation, not background-only motion.

12. **VideoCapture Resource Leak on Exceptions (Resolved - May 08)**:
    - **Problem**: `_extract_ego_motion` and `_extract_and_correlate_pose` both opened `cv2.VideoCapture` instances and relied on a single `cap.release()` call at the end of the happy path. If any intermediate operation raised — `cv2.calcOpticalFlowFarneback` on a malformed frame, `self.model(crop, ...)` failing under MPS memory pressure, `np.percentile(mag, 95)` on an empty array, or any indexing error during keypoint extraction — control jumped to the outer `try/except` in `run`, and the `cap` object was leaked until garbage-collected.
    - **Solution**: Both methods now wrap their post-`cap.set(...)` logic in `try: ... finally: cap.release()`, so the capture is released regardless of exception path. The duplicated early-return `cap.release()` calls were removed since `finally` handles them uniformly. Resolved Issue 5's `cap.isOpened()` and `fps == 0` guards remain at the top of each method (before the `try` block) since they must run before any state needs cleanup.

13. **YOLOv8 First-Detection Selection Without Bystander Filtering (Resolved - May 08)**:
    - **Problem**: The pipeline ran YOLOv8-pose on the cropped bystander bbox and unconditionally selected `results[0].keypoints.data[0]` — the first person detection. When the crop contained multiple persons (overlapping bystanders, partial third person at the bbox edge, or a YOLO false positive), keypoint index 0 could correspond to a different person than the intended bystander, invalidating downstream velocity and spine-angle computations across consecutive frames — the same failure mode Resolved Issue 2 fixed for keypoint *indices* but not for *person* selection.
    - **Solution**: Added a `_select_pose_detection(results, crop_w, crop_h)` helper that computes IoU between each detection's bbox (`results[0].boxes.xyxy[i]`) and the original input crop reference `(0, 0, crop_w, crop_h)`, picking the detection with the highest IoU as the intended bystander. When `boxes` is unavailable (older ultralytics versions, malformed responses), the helper falls back to the largest keypoint-spatial-spread heuristic. The replacement is invoked in `_extract_and_correlate_pose` in place of `results[0].keypoints.data[0]`, preserving cross-frame person identity within a single bystander bbox.

14. **YOLO No-Detection Guard Checked the Wrong Tensor Dimension (Resolved - May 08)**:
    - **Problem**: The guard `if not results or not results[0].keypoints or results[0].keypoints.data.shape[1] == 0: continue` inspected `shape[1]` — the keypoint-count dimension (always 17 for a successful pose model) — instead of `shape[0]`, the person-count dimension. When YOLO returned zero person detections inside the bystander crop (small/occluded crop, motion blur, or pose-confidence below internal threshold), `data` had shape `(0, 17, 3)`. `shape[1]` was still 17, so the guard evaluated False, control fell through to indexing `data[0]`, raising `IndexError: index 0 is out of bounds for axis 0 with size 0`. The exception was caught by the outer `try/except` in `run`, aborting the *entire* video over a single empty-frame detection.
    - **Solution**: The new `_select_pose_detection` helper checks `r.keypoints.data.shape[0] == 0` (person count) and returns `None` to signal "no detection in this frame." It additionally guards against malformed tensors with `shape[1] < 17`. The caller in `_extract_and_correlate_pose` now `continue`s on `None`, so empty-frame detections skip cleanly without aborting the rest of the reaction window.

15. **Mirroring Detection Inherits the Chaos-Floor False-Negative (Resolved - May 08)**:
    - **Problem**: Resolved Issue 3 added an absolute chaos floor (`max_chaos_score < 3.0`) that returned an empty spike list from `_extract_ego_motion`, suppressing every ego spike. Downstream, `_correlate_mirroring` iterated `ego_spikes` to gate detection. When the spike list was empty, mirroring was never evaluated. However, the documented intent of mirroring — "If the EgoMotion pans down rapidly (POV person leaning over), and the bystander's spine keypoints angle inward congruently" — does not require *chaotic* camera motion. A calm, deliberate downward camera tilt while the parent leans toward a fallen child is exactly the scenario the layer should catch, yet it produced no chaos spike, no ego_spike, and therefore no mirroring detection. The chaos-floor fix correctly suppressed flinch false-positives but inadvertently suppressed legitimate mirroring true-positives.
    - **Solution**: `_extract_ego_motion` now returns three parallel signals: `(chaos_spikes, vertical_flow_timeline, norm_max_chaos)`. `chaos_spikes` is still subject to the 3.0 floor for flinch correlation; `vertical_flow_timeline` is the full unfiltered `[(t, mean_v), ...]` series and is always populated. `_correlate_mirroring` now consumes `vertical_flow_timeline` directly, scanning for sustained `v_flow < MIRRORING_VFLOW_THRESHOLD` regions independent of the chaos floor. The flinch false-positive suppression remains intact (chaos-floor still gates the spike list); calm-but-tilting POV footage now correctly produces mirroring detections when the bystander's spine angle changes congruently. `process_video`'s sentinel logic was updated so a video with no chaos spikes but a populated vertical-flow timeline still proceeds to per-bystander pose evaluation rather than short-circuiting on `no_ego_spikes`.

16. **`prev_kpts_by_idx` Reset Broke Velocity Chain on Sparse Detection Failures (Resolved - May 08)**:
    - **Problem**: The pose extraction loop assigned `prev_kpts_by_idx = current_kpts_by_idx` unconditionally, even when `current_kpts_by_idx` was empty (every relevant keypoint fell below the 0.5 confidence threshold for that frame). On the next iteration, `prev_kpts_by_idx` was `{}`, the `common_keys` intersection with the new frame was empty, and no velocity was recorded. Velocity computation only resumed after two *consecutive* frames with high-confidence keypoints. For bystander reaction windows where motion blur, partial occlusion (e.g., a hand passing in front of the face during a flinch), or YOLO confidence dips intermittently dropped frames, the velocity chain repeatedly reset, and the peak velocity from the actual flinch could be missed entirely.
    - **Solution**: `prev_kpts_by_idx` and `prev_t` are now only updated when `current_kpts_by_idx` is non-empty — a failed-detection frame leaves the prior reference in place so the next successful frame can still compute velocity. To prevent stale comparisons across long occlusions, the velocity computation is gated on `0 < dt <= PREV_KPTS_CARRY_FORWARD_SEC` (0.5 s); gaps longer than the cap silently skip that velocity sample and wait for the next pair. Velocities computed across a small gap use the actual elapsed `dt`, so larger gaps yield naturally lower magnitudes — slight signal attenuation rather than total loss, which aligns with the spirit of Resolved Issue 2's keypoint-index fix.

17. **YOLOv8 Pose Default Promoted From `yolov8n-pose` to `yolov8x-pose` (Resolved - May 15)**:
    - **Problem**: Resolved Issue #1 substituted MMPose/RTMPose with `ultralytics` YOLOv8-pose using the **`yolov8n-pose.pt`** (nano, ~6.5 MB) variant. The nano tier was selected against the 24 GB Mac mini M4 Pro budget, where the resident set across L2CS-Net (03a), Py-Feat (03b), emotion2vec+ (03c), and Depth Anything V2 + SAM ViT-Base (03d) left no room for a larger pose backbone. Nano's keypoint accuracy on partially-occluded bystanders — the dominant failure mode for `bystander_pose_velocity_peak` — was the limiting factor for the flinch metric on the new **Mac Studio (M4 Max, 64 GB unified memory)** host, where ~140 MB for the largest variant is loadable with no displacement of any other 03 layer.
    - **Solution**: Switched the default model identifier in `MotorResonancePipeline._init_model` (`src/layer_03f_motor_resonance/pipeline.py`) from `yolov8n-pose.pt` to `yolov8x-pose.pt` (both the on-disk lookup `base_dir / "yolov8x-pose.pt"` and the ultralytics-download fallback string). The weights file `yolov8x-pose.pt` is already present in the project root. Selected the `x` tier directly per the chosen remediation path; gain over `m`/`l` is marginal on the bystander-crop input size that 03f actually uses, but the largest variant is the furthest insurance against partial-occlusion false-negatives and remains well within the M4 Max budget. All 8 tests in `tests/test_layer_03f.py` pass, including `test_keypoint_alignment_uses_dict_not_list` (Resolved Issue #2 fixture) and `test_output_schema_on_synthetic_video` (Resolved Issue #4 fixture); the existing COCO keypoint schema (indices 5/6/9/10/11/12) is unchanged, so the keypoint-dict alignment and crop-diagonal normalization logic continues to apply identically. `KPT_CONFIDENCE_THRESHOLD = 0.5` (Resolved Issue #8) was not recalibrated for the larger model's confidence distribution; if downstream evaluation shows systematic under- or over-confidence, that constant is the single-site tuning knob.
    - **Follow-up (May 16)**: The hardcoded `yolov8x-pose.pt` path has been replaced by `get_model("layer_03f_pose")` from the tier-per-host registry (`ml_dependencies.md` Resolved Issue #1). The `medium`/`large` tiers still resolve to `yolov8x-pose.pt`; the `small` tier resolves back to `yolov8n-pose.pt` for the 24 GB Mac mini fallback. Override via `SR_MODEL_TIER`.

18. **RTMPose Re-Evaluation Deferred to `ml_dependencies.md` Verdict (Resolved - May 15)**:
    - **Problem**: Resolved Issues #1 and #6 dropped `MMPose`/`RTMPose` because `mmcv` compilation failed on Apple Silicon (M4 Pro). On the M4 Max with a newer Clang baseline and current `mmcv-full` releases the compilation chain may now succeed. RTMPose-l outperforms YOLOv8x-pose on COCO-keypoints by ~3 AP, which is potentially material for the flinch-velocity primary metric. The compilation-status question is owned by `ml_dependencies.md` Unresolved Issue 2 ("RTMPose / MMPose Compilation Status on M4 Max Unverified"), so opening a parallel A/B investigation at the 03f layer before that question is answered would duplicate the build-chain work.
    - **Solution**: No code change in `src/layer_03f_motor_resonance/`. Decision recorded: the RTMPose-vs-YOLOv8x A/B is gated on `ml_dependencies.md` Issue 2 resolving with a working `mmcv` install on the M4 Max. If that issue closes successfully, a fresh issue will be filed in this doc with a concrete RTMPose-l vs YOLOv8x-pose A/B against the existing 03f test fixtures (`tests/test_layer_03f.py` plus the Resolved Issue #2 and #4 fixtures). If `ml_dependencies.md` Issue 2 closes with `mmcv` still failing, RTMPose stays permanently dropped and `yolov8x-pose.pt` (Resolved Issue #17) remains the production model. Keeping the decision-making in dependency-order ensures the model choice is benchmark-driven rather than speculative.
    - **Closure note (May 16)**: `ml_dependencies.md` Issue 2 closed without re-attempting the `mmcv` build — the MMPose / RTMPose rows were deleted from both the Local Models and Core Libraries tables as permanent removals (see `ml_dependencies.md` Resolved Issue #2). Per the conditional above, `yolov8x-pose.pt` is now the only production pose backbone in `03f`; no further RTMPose A/B is planned.

