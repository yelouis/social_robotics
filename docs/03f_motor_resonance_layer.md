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
- **Bystander Masking (vertical flow)**: The mean vertical-flow signal that the Mirroring Metric consumes is computed over *background pixels only* — `_bystander_mask_for_frame` zeroes out the nearest-timestamp bbox of every bystander, so a large in-frame bystander's own motion cannot masquerade as a camera tilt. (The chaos percentile is also masked over background pixels — see Resolved #7.)

### 2. Bystander Pose Extraction (YOLOv8 Pose)
We must track how the bystander responds physically. 
- **Recommended SOTA Toolkit**: Use the **Ultralytics YOLO** framework, loading the **YOLOv8-pose** model architecture via the tier-per-host registry (`get_model("layer_03f_pose")`, defaulting to `yolov8x-pose.pt`). The `x` tier maximizes keypoint accuracy on partially-occluded bystanders, which is the dominant failure mode for flinch metrics. It replaces earlier MMPose/RTMPose specifications to avoid complex `mmcv` compilation failures on Apple Silicon.
- **Mechanism**: Run YOLOv8-pose on the bystander's bounding box during the reaction window. It natively supports PyTorch MPS tensors on Apple Silicon (validated on the **Mac Studio M4 Max** target host) for target real-time FPS.
- **Bystander-anchored window + dense pose sampling**: Node-02 bystander detections are ~6 s-cadenced and sit a median ~10 s from the wearer's climax, so the strict reaction window usually held only 0–1 detections and velocity (which needs ≥ 2 pose frames) was always 0 (the layer scored 0/50). The pose window is therefore anchored via the shared `bystander_measurement_window` helper (`src/shared/bystander_window.py`; `min_in_reaction_window=1`, capped at `MAX_ANCHOR_SPAN_SEC`), and YOLO-pose is sampled **densely at `TARGET_POSE_FPS = 10`** across it via a sequential `grab()`/`read()` walk, with the bystander bbox **interpolated between the sparse Node-02 detections** (`_interp_bbox`). *(Rationale + validation: Resolved #8.)* Each task expands into one segment per bystander cluster (multi-window — docs/03 § Multi-Window Reaction Segments), so 03f applies the cross-layer per-segment guardrails (**dedupe** segments that re-anchor to the same `(person, measurement_window)`; untracked/over-sparse tracks are already excluded by 03f's existing `MIN_GENUINE_DETECTIONS = 2` dense-track gate, Resolved #10) so a phantom track is not counted once per segment.

### 3. Correlating the Resonance
- **The Flinch Metric**: Calculate the velocity of the bystander's wrist/shoulder keypoints. If they rapidly elevate (throwing hands up defensively) within `0.5s` of a spike in POV Ego-Kinetic Energy, we have detected a sympathetic physical flinch.
- **Scale-Invariant Velocity**: Keypoint coordinates are divided by the crop's diagonal length at inference time, converting positions into a `[0, 1]` space before differencing. This is load-bearing: raw crop-local pixels made the same physical arm movement produce drastically different "velocities" depending on bystander distance/crop size; the normalizer (`VELOCITY_NORMALIZER = 0.5` diagonals/sec ≈ a significant flinch) is calibrated for the normalized space and must not be re-pointed at raw pixels.
- **Mirroring Metric**: If the EgoMotion pans down rapidly (POV person leaning over), and the bystander's spine keypoints (shoulder to hip) angle inward congruently, they are physically mirroring the intention.
- **False-positive gating (impulse + dense-track)**: a raw "velocity spike within 0.5 s of an ego spike" correlation over-fires on coincidental or camera-induced motion, so `motor_resonance_detected` is guarded by two gates: (a) the ego jolt must be an **impulsive, dominant** jolt rather than sustained or coincidental window motion (Resolved #9); and (b) pose velocity is trusted **only when ≥ 2 genuine Node-02 detections** fall inside the anchored window (`MIN_GENUINE_DETECTIONS = 2`) — with a single carried/interpolated box a camera pan drags the scene through a fixed bbox and YOLO reports apparent keypoint velocity time-locked to the jolt *by construction* (Resolved #10). The Mirroring Metric is a separate signal and is deliberately left ungated.
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
   - **Solution**: Added a `TARGET_FLOW_FPS = 10.0` class constant. `_extract_ego_motion` computes `frame_stride = max(1, round(fps / TARGET_FLOW_FPS))` and advances in stride-sized hops. At 30 FPS this is a 3× speedup with negligible loss in jolt detectability. *(Note: the stride was originally advanced via per-step `cap.set` seeking — since replaced by a sequential `grab()`/`read()` walk, see Resolved #6.)*

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
   - **Solution**: Replaced the per-step seek with the repo-standard sequential `grab()`/`read()` skip (grab forward to the target frame, read only the analyzed frame), matching `shared/climax_extraction.py` and 03a. **Validated bit-identical** on 3 real cached clips: spike count, spike timestamps, `vertical_flow_timeline`, and `ego_kinetic_chaos_score` were all unchanged vs the pre-fix output. The sparse per-detection seeks in `_extract_and_correlate_pose` are intentionally left (widely-spaced samples are the correct use of seeking, per the face-quality-prefilter precedent).

7. **Ego Chaos Score Not Bystander-Masked — Self-Correlation False Positive (Resolved - June 12)**:
   - **Problem**: Found in the June 12 repo audit. The chaos metric (`np.percentile(mag, 95)`) was computed over the **full downsampled frame** while only the vertical-flow (mirroring) signal applied `_bystander_mask_for_frame`. This let a large in-frame bystander's own abrupt movement raise the "ego" chaos, enabling a **self-correlation false positive**: the bystander's flinch creates the chaos spike, then `_extract_and_correlate_pose` detects that same flinch within `RESONANCE_WINDOW_SEC` of "the wearer's" spike → `motor_resonance_detected: true` with no actual wearer event, inverting the flinch metric's premise.
   - **Solution** (Option A): The chaos percentile now uses the same `_bystander_mask_for_frame` as `mean_v` — `np.percentile(mag[mask], 95)` over background pixels, with the existing full-frame fallback when the mask is empty. On the 3-clip probe this removed the expected bystander-driven spurious spikes (e.g. 27→26, 36→30 spikes) while leaving clips with no large bystander unchanged; the chaos floor/normalizer (3.0/20.0) held without re-calibration on this sample. The Implementation Strategy's "EgoMotion = POV-actor motion" claim is now accurate.

8. **Motor Resonance Scored 0/50 — Bystander Pose Was Sampled Only at Sparse In-Window Detections (Resolved - June 15)**:
   - **Problem**: The June 14 smell test (`e2e_reports/2026_06_14_layer03f_50/`) scored **0/50** (0 resonance, 0 mirroring); 44/50 were `no_pose_data` and the 6 "scored" clips had `bystander_pose_velocity_peak = 0.0`. `_extract_and_correlate_pose` extracted pose only at the Node-02 detection timestamps inside the strict `task_reaction_window_sec`, but detections are ~6 s-cadenced and a median ~10 s from the wearer climax, so a window held 0–1 detections — and velocity needs **≥ 2** consecutive keypoint frames. The same wearer-vs-bystander window mismatch as 03b/03d/03e; 03f never received it.
   - **Solution** (Option A): `_extract_and_correlate_pose` now (a) anchors the pose window via the shared `bystander_measurement_window` helper (`src/shared/bystander_window.py` — keep the reaction window when the bystander is detected there, else anchor to the climax-nearest detection, capped at 30 s), and (b) samples YOLO-pose **densely at `TARGET_POSE_FPS` (10)** across that window via a sequential `grab()`/`read()` walk, interpolating the bystander bbox between sparse detections (`_interp_bbox`). **Validated E2E on the 50-clip re-run (`e2e_reports/2026_06_14_layer03f_50_dense/`): scored 6 → 34, motor_resonance 0 → 9, mirroring 0 → 6, `bystander_pose_velocity_peak` 0.0 → median 4.65** — velocity is now computable wherever the bystander is visible near the jolt. 03f is the helper's third consumer (03d/03e). Covered by `test_interp_bbox_midpoint_and_carry` + the updated `_extract_and_correlate_pose` signature test; suite 194/194. **NB — velocity-computation is fixed, but the now-produced resonances are largely false positives** (spot-check: `43bd06f3` pid0 = a man eating; `599f2f09` pid0 = festival-crowd identity jitter — both coincidentally within 0.5 s of an ego spike, not flinches). Resonance *trustworthiness* is filed as the new Unresolved Issue 1 (→ Resolved #9, #10).

9. **Motor Resonance Fired on Coincidental / Sustained Motion — Impulse + Dominant-Jolt Gate (Resolved - June 15)**:
   - **Problem**: After Resolved #8 made velocity computable, 03f flagged **9/34 motor_resonance** — largely false positives (spot-checked: `43bd06f3` = a man eating; `599f2f09` = festival crowd), because (a) dense sampling captures *all* bystander motion (eating/walking), not just flinches, and (b) the egocentric camera spikes almost constantly, so a velocity peak within `RESONANCE_WINDOW_SEC` of *some* spike is near-certain (a multiple-comparisons coincidence).
   - **Solution** (Option A): Extracted `_resonance_decision`, which now requires (a) an **impulsive** flinch — `max_vel >= VELOCITY_NORMALIZER` AND `max_vel >= RESONANCE_IMPULSE_RATIO` (3.0) × the bystander's *own median* velocity (sustained eating/walking is high-but-flat → ratio ~1 → rejected; the median subtracts each bystander's activity baseline), and (b) correlation with only the single **dominant** ego spike (max chaos `score`, now carried on each spike dict), not the whole relative-threshold list. **Validated on the gated 50-clip re-run (`e2e_reports/2026_06_15_layer03f_50_gated/`): motor_resonance 9 → 2 — the eating (`43bd06f3`) and crowd (`599f2f09`) false positives are now `resonance=False`; scoring/velocity is unchanged (34 scored).** Covered by `test_resonance_decision_rejects_sustained_and_nondominant`; suite 195/195. **NB — the 2 survivors are still FPs of a subtler kind** (camera motion through a fixed/sparse bbox), filed as the new Unresolved Issue 1 — now resolved below.

10. **Residual Resonance FPs — Camera Motion Through a Fixed / Sparse Bystander Bbox (Resolved - June 15)**:
   - **Problem**: The 2 survivors of Resolved #9's impulse + dominant-jolt gate were a subtler FP class. Both were **single / sparse-detection** bystanders whose interpolated bbox is effectively constant: `66d4121f` pid16 (**1 detection** @ 33.0 s) and `0235dafb` pid0 (6 detections, median 12 s gaps). When only one *genuine* Node-02 detection lands inside the anchored measurement window, `_interp_bbox` carries that single box across the entire dense-sampled window, so as the wearer's camera pans during the jolt the *scene* shifts through the fixed box and YOLO-pose reports apparent keypoint "velocity" that is time-locked to the jolt **by construction** — a camera-motion self-correlation, not the bystander's own body motion. Distinct from Resolved #7 (which masked the bystander *out of* the ego chaos); here camera motion leaks *into* the bystander pose velocity.
   - **Solution** (Option A): Added the static `_has_dense_track` helper and a `MIN_GENUINE_DETECTIONS = 2` class constant. `_extract_and_correlate_pose` now computes pose velocity only when ≥ 2 **genuine** detections fall inside the anchored window `[w_start, w_end]` (counting the real Node-02 detections, *not* the dense interpolated samples); sparser tracks emit `velocity_peak = 0.0` / `resonance = False` — honest, since a single carried/over-interpolated box is no real track to make a claim from. Mirroring (spine-angle correlation) is a separate signal and is deliberately left ungated. **Validated on the 50-clip re-run (`e2e_reports/2026_06_15_layer03f_50_dense_track/`): motor_resonance 2 → 0; scored 34 and mirroring 6 unchanged.** A per-`(video, person)` diff against the gated run shows the change is surgical: the **9** rows that changed are *all* single-in-window-detection tracks (`genuine_in_window = 1`, incl. both survivors flipping `True → False`), while **63** dense-track rows keep their velocity bit-identical — zero collateral. An offline replay of `bystander_measurement_window` + the gate predicted both survivors' rejection before the GPU run. Covered by `test_has_dense_track_gates_single_and_sparse`; suite 195/195. **NB — a sparse-but-multi track (≥ 2 in-window detections whose box still drifts over background) is not caught by a count gate; Option B (ego-motion-compensate the keypoint velocity via the existing Farneback flow) remains the path if that class surfaces.**

11. **Bystander Track-Explosion Cap + Distinct-Window Dedup (Resolved - June 27)**:
   - **Problem**: 03f had the per-segment loop + the `MIN_GENUINE_DETECTIONS` dense-track gate (Resolved #10) + the anchor bound, but **neither** the track-explosion cap **nor** the distinct-window dedup. Node-02 fragments bystander tracking into many short spurious positive-id tracks — median 48, up to **829/clip** (03d Resolved #6) — and 03f's per-bystander YOLO-pose (dense @ 10 FPS) would process all of them, making the full run infeasible. Separately, a sparse track's segments re-anchor onto the same pose window, so the same flinch would be counted once per segment.
   - **Solution**: `MAX_BYSTANDERS_PER_CLIP = 10` (mirrors 03d) — process only the N longest tracks per clip (the genuine sustained bystanders a scene actually has). Plus a per-clip **distinct-window dedup** — a `(person, window)` set, the cheap `bystander_measurement_window` helper re-derived before the pose call. **Validated** on the full top-200 run via the shared **parallel harness** (`tools/run_parallel_layer.py`, N=3): **200 clips, 0 errors, max 10 persons/clip**, 65 motor-resonance + 402 mirroring events, **~2.5 h** wall (**~3×** vs serial), stress-tested at N=3 with no crash and 89% RAM free. Suite 11/11.

## ⚠️ Unresolved Issues & Suggestions

_No open issues. The camera-motion-through-bbox FP class is **Resolved #10** (dense-track gate); the track-explosion infeasibility is **Resolved #11** (per-clip cap). If a sparse-but-multi-detection drift class later surfaces, **Option B** (ego-motion-compensate the keypoint velocity via the existing Farneback flow) is the documented next step._

