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

6. **Per-Stride `cap.set` Seeking in `_extract_ego_motion` (Resolved - June 12)**:
   - **Problem**: Found in the June 12 repo audit. `_extract_ego_motion` seeked with `cap.set(CAP_PROP_POS_FRAMES, …)` on **every stride step** (~10 Hz across the reaction window). On H.264 `cap.set` is keyframe-approximate — each seek re-decodes from the nearest keyframe (~10–20× wasted decode) and can desync frame↔timestamp, the exact failure mode that corrupted 03a scores before its no-seek rewrite (03a Resolved #5; same fix validated bit-identical in 02 Resolved #18).
   - **Solution**: Replaced the per-step seek with the repo-standard sequential `grab()`/`read()` skip (grab forward to the target frame, read only the analyzed frame), matching `shared/climax_extraction.py` and 03a. **Validated bit-identical** on 3 real cached clips: spike count, spike timestamps, `vertical_flow_timeline`, `ego_kinetic_chaos_score`, and the 03g shift vector were all unchanged vs the pre-fix output. The sparse per-detection seeks in `_extract_and_correlate_pose` are intentionally left (widely-spaced samples are the correct use of seeking, per the face-quality-prefilter precedent).

7. **Ego Chaos Score Not Bystander-Masked — Self-Correlation False Positive (Resolved - June 12)**:
   - **Problem**: Found in the June 12 repo audit. The chaos metric (`np.percentile(mag, 95)`) was computed over the **full downsampled frame** while only the vertical-flow (mirroring) signal applied `_bystander_mask_for_frame`. This let a large in-frame bystander's own abrupt movement raise the "ego" chaos, enabling a **self-correlation false positive**: the bystander's flinch creates the chaos spike, then `_extract_and_correlate_pose` detects that same flinch within `RESONANCE_WINDOW_SEC` of "the wearer's" spike → `motor_resonance_detected: true` with no actual wearer event, inverting the flinch metric's premise.
   - **Solution** (Option A): The chaos percentile now uses the same `_bystander_mask_for_frame` as `mean_v` — `np.percentile(mag[mask], 95)` over background pixels, with the existing full-frame fallback when the mask is empty. On the 3-clip probe this removed the expected bystander-driven spurious spikes (e.g. 27→26, 36→30 spikes) while leaving clips with no large bystander unchanged; the chaos floor/normalizer (3.0/20.0) held without re-calibration on this sample. The Implementation Strategy's "EgoMotion = POV-actor motion" claim is now accurate.

8. **Motor Resonance Scored 0/50 — Bystander Pose Was Sampled Only at Sparse In-Window Detections (Resolved - June 15)**:
   - **Problem**: The June 14 smell test (`e2e_reports/2026_06_14_layer03f_50/`) scored **0/50** (0 resonance, 0 mirroring); 44/50 were `no_pose_data` and the 6 "scored" clips had `bystander_pose_velocity_peak = 0.0`. `_extract_and_correlate_pose` extracted pose only at the Node-02 detection timestamps inside the strict `task_reaction_window_sec`, but detections are ~6 s-cadenced and a median ~10 s from the wearer climax, so a window held 0–1 detections — and velocity needs **≥ 2** consecutive keypoint frames. The same wearer-vs-bystander window mismatch as 03b/03d/03e; 03f never received it.
   - **Solution** (Option A): `_extract_and_correlate_pose` now (a) anchors the pose window via the shared `bystander_measurement_window` helper (`src/shared/bystander_window.py` — keep the reaction window when the bystander is detected there, else anchor to the climax-nearest detection, capped at 30 s), and (b) samples YOLO-pose **densely at `TARGET_POSE_FPS` (10)** across that window via a sequential `grab()`/`read()` walk, interpolating the bystander bbox between sparse detections (`_interp_bbox`). **Validated E2E on the 50-clip re-run (`e2e_reports/2026_06_14_layer03f_50_dense/`): scored 6 → 34, motor_resonance 0 → 9, mirroring 0 → 6, `bystander_pose_velocity_peak` 0.0 → median 4.65** — velocity is now computable wherever the bystander is visible near the jolt. 03f is the helper's third consumer (03d/03e). Covered by `test_interp_bbox_midpoint_and_carry` + the updated `_extract_and_correlate_pose` signature test; suite 194/194. **NB — velocity-computation is fixed, but the now-produced resonances are largely false positives** (spot-check: `43bd06f3` pid0 = a man eating; `599f2f09` pid0 = festival-crowd identity jitter — both coincidentally within 0.5 s of an ego spike, not flinches). Resonance *trustworthiness* is filed as the new Unresolved Issue 1.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: `motor_resonance_detected` Fires on Coincidental Bystander Motion Near Abundant Ego Spikes (False Positives)
**Status**: ⚠️ Confirmed Unresolved — Exposed by the June 15 dense-pose run (Resolved #8). With velocity now computable, 03f flags **9/34 motor_resonance** and **6/34 mirroring** — implausibly high for genuine mirror-neuron flinches on egocentric footage. Spot-checking the top detections shows ordinary motion coincidentally time-locked to a spike, not stimulus-locked flinches: `43bd06f3` pid0 (vel 4.99, resonance, empathy 1.0) is a man **eating** (hand-to-mouth); `599f2f09` pid0 (vel 7.55) is a **festival crowd** where the bbox jumps between bodies. Three compounding causes: (1) dense sampling captures *all* bystander motion (eating, walking, fidgeting), not just flinches; (2) the egocentric camera spikes almost constantly (`ego_kinetic_chaos_score` 0.97–1.0 on most clips), so a velocity peak within `RESONANCE_WINDOW_SEC` (0.5 s) of *some* spike is near-certain for any moving bystander (a multiple-comparisons coincidence); (3) ego spikes are measured over the strict reaction window while pose is measured over the (often offset) bystander-anchored window, so the "stimulus → response" pairing can be temporally incoherent. The flinch premise — a defensive response time-locked to a *specific severe* jolt — is not actually being tested.

**Option A (recommended)**: **Dominant-jolt gate + impulsive, baseline-relative flinch.** Gate resonance on a single clearly-dominant chaos spike (not the relative-threshold list that fires on many), and require the bystander velocity to be a sharp impulse well above that bystander's *own* median velocity (subtract their activity baseline), since eating/walking are sustained, not impulsive.
  - *Pros*: Directly attacks the eating/walking false positives (baseline subtraction) and the spike-abundance coincidence (dominant-spike gate); no new model.
  - *Cons*: Drops yield toward the honest low rate of true flinches; several thresholds to tune; calibration needs scarce flinch footage.

**Option B**: **Require ego/pose temporal coherence.** Only attempt resonance when the bystander-anchored pose window overlaps the reaction window (bystander present *during* the jolt); otherwise emit velocity/mirroring but force `motor_resonance_detected=False` (no co-located stimulus).
  - *Pros*: Removes temporally-incoherent pairings; cheap; honest.
  - *Cons*: Doesn't fix eating/crowd false positives within an overlapping window; reduces yield.

**Option C**: **Demote resonance to a provenance-tagged candidate.** Keep emitting it but add the ego-spike count and a baseline-relative velocity ratio so Layer 04 can filter; treat raw `motor_resonance_detected` as low-confidence.
  - *Pros*: Lossless; defers policy downstream; no tuning now.
  - *Cons*: Pushes the false-positive problem to 04; the boolean stays misleading alone.

Your selection: _____

