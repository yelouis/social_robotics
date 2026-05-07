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

6. **RTMPose / MMPose Native Integration (Resolved - May 05)**:
   - **Problem**: The pipeline originally specified RTMPose and MMPose, but `mmcv` compilation fails on Apple Silicon (M4 Pro). The documentation (`ml_dependencies.md`) still listed RTMPose as a requirement, creating a discrepancy between the codebase and specifications.
   - **Solution**: Formally updated `ml_dependencies.md` to designate `ultralytics` YOLOv8-pose (`yolov8n-pose.pt`) as the primary production model and marked RTMPose as Deferred/Optional. This eliminates misleading dependency documentation and standardizes around the functional PyTorch MPS backend.

7. **Mirroring Metric Not Implemented (Resolved - May 05)**:
   - **Problem**: The original specification called for tracking congruent spine-angle changes between the POV camera and the bystander. The implementation lacked the ability to extract hip keypoints, compute spine angles, and correlate them with vertical EgoMotion flow.
   - **Solution**: Expanded keypoint extraction in `pipeline.py` to include COCO hip indices (11/12). Added a `_correlate_mirroring` method that calculates the spine angle (`atan2`) and correlates significant inward angle changes (>0.1) with downward EgoMotion spikes (`v_flow < -1.0`) within the reaction window. Output updated to include `mirroring_detected` and `mirroring_scalar`.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Skipped Videos Reprocessed on Every Resume
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:99-117`. When `process_video` returns `None` (file missing at line 127, no bystanders/tasks at line 134, no qualifying ego spikes at line 151, all bystanders yield no `pose_velocities` at line 367, or every reaction window is filtered out), the `if result:` guard skips both `results.append(...)` and `self.processed_ids.add(video_id)`. Subsequent resume runs re-iterate these videos, re-execute Farneback optical flow over the entire reaction window, and re-run YOLOv8-pose inference per sampled frame, only to discard the result again. For batches with many edge-case videos (e.g., bystanders out-of-frame during the reaction window, or calm videos that fail the chaos floor), this consumes substantial MPS time on guaranteed no-op work. Errors caught by the `except` branch at line 116 similarly never mark `video_id` processed.

**Option A (recommended)**: **Sentinel-Record Tracking** — When `process_video` returns `None`, write a sentinel record (e.g., `{"video_id": ..., "layer": "03f_motor_resonance", "tasks_analyzed": [], "skipped_reason": "no_ego_spikes" | "no_pose_data" | "no_bystanders"}`) to results and mark the id processed.
  - *Pros*: Persists skip decisions; downstream consumers gain visibility into why a video was excluded; resume cost drops to O(JSON-load + set-membership).
  - *Cons*: Output JSON inflates with empty entries; downstream consumers must filter on `tasks_analyzed` length or `skipped_reason`.

**Option B**: **Skip Manifest Sidecar** — Maintain `03f_skipped.json` listing video_ids that produced no output, short-circuit them at the top of the per-entry loop alongside `processed_ids`.
  - *Pros*: Keeps the main result JSON clean; explicit skip log is easy to audit.
  - *Cons*: Two-file state machine to maintain; risk of skip-manifest/result-manifest drift if writes aren't atomic.

**Option C**: **Always-Mark-Processed Policy** — Always add `video_id` to `processed_ids` after `process_video` returns regardless of value, persisting `processed_ids` to disk separately.
  - *Pros*: Cleanly decouples "did we attempt this?" from "did it produce output?"; minimal JSON inflation.
  - *Cons*: Adds a third state file; tests must mock or write the new file; loses the rationale for *why* a video was skipped.

Your selection: _____

---

### Issue 2: `_extract_ego_motion` Reads Every Source Frame at Full FPS
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:204-238`. The method seeks to `start_frame = int(start_sec * fps)` and increments `current_frame_idx` by 1 in every iteration (`current_frame_idx += 1` at line 238), reading and Farneback-flowing *every* frame in the reaction window. For a 30 FPS source and a 5-second reaction window, this is 150 dense optical-flow computations per task per video, which dominates wall-clock per video and scales poorly. Sister layer 03d's `_extract_ego_motion_noise` (in `src/layer_03d_proxemic_kinematics/pipeline.py`) follows the same pattern, so the cost compounds across the pipeline. The pipeline already downsamples each frame to 0.5x resolution (lines 215, 226), but does not subsample temporally.

**Option A (recommended)**: **Adaptive Frame Stride for Optical Flow** — Skip frames so that effective flow-FPS is ~10 Hz (e.g., `frame_stride = max(1, int(fps / 10))`), iterating `current_frame_idx += frame_stride`. The `dt` between consecutive `prev_gray` and `gray` is correspondingly larger; multiply optical-flow magnitude by `frame_stride` so the chaos score remains comparable to per-frame magnitude.
  - *Pros*: 3x speedup on 30 FPS videos with negligible loss in jolt detectability (jolts last 100-300ms and are easily caught by 10 Hz sampling); preserves the existing chaos-score scale via the multiplier; no schema change.
  - *Cons*: Borderline jolts shorter than 100ms could be missed; requires the multiplier to be calibrated; sister layer 03d would need a parallel update for consistency.

**Option B**: **Cache Ego-Motion Per (video_id, window) Tuple** — Persist computed `chaos_scores` lists keyed by video_id and reaction window to a sidecar JSON. On re-run or resume, load from cache if hit.
  - *Pros*: Eliminates repeat cost on re-runs entirely; useful during ablation experiments where pose detection is being tweaked.
  - *Cons*: Adds disk I/O and cache-invalidation logic; first run sees no benefit; cache size grows with batch.

**Option C**: **Batch Frame Reads via `cv2.VideoCapture.read()` Streaming** — Replace per-frame `cap.set(...)` seeking + read with a streaming read loop that pre-buffers frames and emits flow lazily.
  - *Pros*: Eliminates seek overhead which can be ~ms-per-frame on long-GOP codecs.
  - *Cons*: More complex control flow; the current code already streams via sequential `cap.read()` after a single seek, so gain is small.

Your selection: _____

---

### Issue 3: `mean_v` Pollution by Foreground Motion in Mirroring Signal
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:230` and `pipeline.py:418-440`. The vertical optical-flow signal is computed as `mean_v = np.mean(flow[..., 1])` over the *entire* downsampled frame, including the bystander's body. The "Mirroring Metric" then uses this signal at `pipeline.py:426` (`if v_flow < -1.0`) to gate detection of "downward EgoMotion (camera pans down)". When the bystander is large in frame (close-up, multiple bystanders, or a leaning person filling 30%+ of pixels), their own vertical motion contaminates the global mean, either fabricating a fake "camera pan down" signal or canceling a real one. The documented intent — "Compute dense Optical Flow on the *background* pixels" — is not realized; the implementation flows the entire frame.

**Option A (recommended)**: **Mask Out Bystander Bounding Boxes Before Averaging** — Before `np.mean(flow[..., 1])`, zero out (or mask) the regions inside all bystander bboxes for the current frame. Compute the mean only over remaining (background) pixels.
  - *Pros*: Aligns implementation with documented intent ("background pixels"); cheap (single mask construction per frame); leverages already-available bystander bbox data.
  - *Cons*: Bbox masks are rectangular and may exclude legitimate background pixels at the edges; requires plumbing bystander bboxes into `_extract_ego_motion`.

**Option B**: **Use the Median Instead of Mean** — Replace `np.mean(flow[..., 1])` with `np.median(flow[..., 1])`. Foreground motion contributes ~30% of pixels in the worst case, so median is robust as long as the bystander occupies <50% of frame.
  - *Pros*: One-line change; no bbox plumbing; resistant to localized motion.
  - *Cons*: Fails when foreground exceeds 50% of frame; biases toward zero in mostly-static frames; loses signal magnitude for legitimate global pans.

**Option C**: **Border-Strip Sampling** — Compute `mean_v` only over the outer 20% border of the frame (top, bottom, left, right strips), which almost always contains background.
  - *Pros*: No bbox plumbing; deterministic background sampling.
  - *Cons*: POV videos often have no clear "edge background" (head-mounted camera filling frame); falsely zeroes the signal in close-quarters scenes.

Your selection: _____

---

### Issue 4: VideoCapture Resource Leak on Exceptions
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:194-259` (`_extract_ego_motion`) and `pipeline.py:261-365` (`_extract_and_correlate_pose`). Both methods open `cv2.VideoCapture` instances and rely on a single `cap.release()` call at the end of the happy path. If any of the intermediate operations raise — `cv2.calcOpticalFlowFarneback` on a malformed frame, `self.model(crop, ...)` failing under MPS memory pressure, `np.percentile(mag, 95)` on an empty array, or any `numpy`/`cv2` error during keypoint extraction — control jumps to the outer `try/except` at `pipeline.py:116` (`run`), and the `cap` object is leaked until garbage-collected. Resolved Issue 5 only added guards for `cap.isOpened()` and `fps == 0` at the *top* of the method, not exception-safe cleanup throughout.

**Option A (recommended)**: **Wrap `cap` in `try/finally`** — Refactor both methods to wrap their post-`cap.set()` logic in `try: ... finally: cap.release()`. Removes the duplicated `cap.release()` call from both early-return paths since `finally` handles them uniformly.
  - *Pros*: Robust to all exception types; idiomatic Python; eliminates duplicated release calls; minimal logic change.
  - *Cons*: Slightly larger indentation diff; need to ensure no return statement bypasses the finally.

**Option B**: **Use `contextlib.closing(cap)` Context Manager** — Wrap `cap` with `contextlib.closing(cv2.VideoCapture(...))` and use a `with` block for the entire method body.
  - *Pros*: Pythonic; explicit lifecycle; mirrors file-handle patterns.
  - *Cons*: `cv2.VideoCapture` doesn't natively implement context-manager protocol; `closing` calls `cap.close()` which doesn't exist (would need `cap.release` shim); slightly awkward.

**Option C**: **Subclass `cv2.VideoCapture` With Context-Manager Support** — Create a `ManagedVideoCapture` helper that supports `__enter__`/`__exit__` and use it across all layers.
  - *Pros*: Reusable across 03a, 03d, 03e, 03f; clean call sites.
  - *Cons*: Touches multiple layers (out of scope for 03f-only audit); adds a shared utility module.

Your selection: _____

---

### Issue 5: YOLOv8 First-Detection Selection Without Bystander Filtering
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:322`. The pipeline runs YOLOv8-pose on the cropped bystander bbox and unconditionally selects `results[0].keypoints.data[0]` — the first person detection. When the crop contains multiple persons (overlapping bystanders, a partial third person at the bbox edge, the POV actor's reflected hand, or YOLO false positives picking up the photographer's body in a mirror), keypoint index 0 may correspond to a different person than the intended bystander. This invalidates downstream velocity, spine-angle, and resonance computations because keypoints across consecutive frames may belong to different individuals — the same failure mode that Resolved Issue 2 fixed for keypoint *indices* but not for *person* selection.

**Option A (recommended)**: **Bbox-IoU Selection** — When YOLO returns multiple person detections in the crop, compute IoU between each detection's bbox (`results[0].boxes`) and the original input crop bbox `(0, 0, crop_w, crop_h)`. Pick the detection with the highest IoU (i.e., the one most aligned with the cropped bystander).
  - *Pros*: Principled selection; preserves cross-frame person identity within a single bystander; cheap.
  - *Cons*: Multiple bystanders close together still ambiguous; relies on YOLO box accuracy.

**Option B**: **Largest-Detection Heuristic** — Pick the detection with the largest bbox area, on the assumption that the intended bystander dominates the crop.
  - *Pros*: One-line change; no IoU computation.
  - *Cons*: Fails when a partial third-party detection is closer to the camera than the bystander; less robust than IoU.

**Option C**: **Run YOLO With Tracking (`model.track`) on the Full Frame** — Replace per-frame crop+detect with `ultralytics`' built-in tracker on the full frame; map detections back to the bystander via bbox IoU with the input bystander bbox.
  - *Pros*: Cross-frame person identity is maintained by the tracker; eliminates the "first detection" ambiguity entirely.
  - *Cons*: Larger refactor; tracker state must be reset per video; full-frame inference is slower than crop inference.

Your selection: _____

---

### Issue 6: Hardcoded Heuristic Constants Throughout the Detection Path
**Status**: ⚠️ Confirmed Unresolved — Magic numbers governing classification decisions are inlined as literals: chaos floor `3.0` at `pipeline.py:253`, max-chaos normalizer `20.0` at `pipeline.py:249`, relative spike threshold `0.7` at `pipeline.py:257`, frame-resize factor `0.5` at `pipeline.py:215, 226`, keypoint confidence threshold `0.5` at `pipeline.py:329`, velocity normalizer `0.5` at `pipeline.py:381`, velocity cap `10.0` at `pipeline.py:381`, reaction-window cap `0.5s` at `pipeline.py:391`, empathy normalizer `5.0` at `pipeline.py:401`, mirroring vertical-flow threshold `-1.0` at `pipeline.py:426`, mirroring time window `±0.5s` at `pipeline.py:427-428`, mirroring-delta threshold `0.1` at `pipeline.py:435`, and mirroring-scalar normalizer `0.5` at `pipeline.py:437`. None are exposed via constructor arguments, configuration, or class-level constants. Tuning the pipeline (e.g., adjusting flinch sensitivity for a different age cohort, recalibrating after a YOLOv8 model swap, or adapting the chaos floor for handheld vs. head-mounted POV) requires source edits.

**Option A (recommended)**: **Class-Level Tuning Constants Block** — Hoist all heuristic constants into a `# --- Detection Tuning ---` block at the top of `MotorResonancePipeline` (e.g., `CHAOS_FLOOR = 3.0`, `CHAOS_NORMALIZER = 20.0`, `SPIKE_RELATIVE_THRESHOLD = 0.7`, `KPT_CONFIDENCE_THRESHOLD = 0.5`, `VELOCITY_NORMALIZER_DIAG_PER_SEC = 0.5`, `VELOCITY_CAP = 10.0`, `RESONANCE_WINDOW_SEC = 0.5`, `EMPATHY_NORMALIZER = 5.0`, `MIRRORING_VFLOW_THRESHOLD = -1.0`, `MIRRORING_TIME_WINDOW_SEC = 0.5`, `MIRRORING_DELTA_THRESHOLD = 0.1`, `MIRRORING_SCALAR_NORMALIZER = 0.5`).
  - *Pros*: Centralizes tuning surface; easy to override via subclassing for ablations; zero runtime overhead; no schema change.
  - *Cons*: Still requires code edit to retune in production; not externally configurable per-batch.

**Option B**: **Config File (YAML/JSON)** — Load constants from a `motor_resonance_config.yaml` co-located with the manifest path.
  - *Pros*: Non-developers can retune; supports per-experiment configurations; auditable as artifacts.
  - *Cons*: Adds a config-loader dependency; tests must construct config fixtures; potential for misconfiguration drift.

**Option C**: **Constructor Arguments with Defaults** — Add named parameters to `__init__` with sensible defaults pulled from current literals.
  - *Pros*: Pythonic; tests can override per-test; type-hints document the tuning surface.
  - *Cons*: Constructor-signature explosion (12+ new args); orchestration code must thread parameters through.

Your selection: _____