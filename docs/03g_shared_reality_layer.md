# AI Task Breakdown: Shared Reality Layer (03g)

## Objective
The **Shared Reality Layer** mathematically evaluates the psychological concept of "Social Referencing." In classic tests (like the Visual Cliff experiment), an infant facing an ambiguous or new situation pauses, turns to the caregiver, and seeks validation. This layer tracks the camera telemetry to see if the POV actor explicitly *verifies* the bystander's reaction after completing a task.

---

## 📥 Input Requirements
- **`filtered_manifest.json`**: For task tracking and `bystander_detections` bounding boxes.
- **Raw Video / Saliency Engine**: To track what the camera is focusing on.

---

## 🛠️ Implementation Strategy

### 1. Identifying the Task Centroid
Use Ego4D task bounding boxes (or center-frame assumption during the `task_climax_sec`) to identify the visual coordinates of the action being performed (e.g., cutting a tomato on the counter). 

### 2. Tracking the POV Pan (Saliency / Optical Flow)
Immediately following the task climax (the start of the `task_reaction_window_sec`), we monitor the macroscopic movement of the camera.
- Using simple Optical Flow (or checking the `bystander_detections` relative coordinates), we watch the panning vector of the frame center.

### 3. The Social Reference Logic
- **Condition A (No Ref)**: The camera remains pointed down at the task (the tomato). The POV wearer is confident and does not care about the social boundary.
- **Condition B (Social Referencing)**: The camera explicitly pans away from the task centroid and centers the bystander's bounding box directly into the middle 30% of the screen. The POV actor is "checking in" with the room.
- *Note:* If the Ego4D dataset contains raw eye-tracking telemetry (often included in Aria glasses datasets), check the `gaze_x, gaze_y` arrays to see if the eye-gaze vector snapped to the bystander's face box.

### 4. Future Possible Direction: Aria Glasses Gaze Telemetry Integration
The current pipeline relies exclusively on optical flow and bounding box centering. Future iterations may explore extracting sub-degree gaze telemetry (`gaze_x`, `gaze_y`) from Aria glasses recordings (`.vrs` formats via `projectaria_tools`). This would provide high-precision ground-truth tracking for social referencing, although it is limited to the Aria-specific subset of clips.

---

## 📤 Output Schema and Integration
**Example Output Data (`03g_shared_reality_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03g_shared_reality",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "post_climax_camera_shift_vector": [450, -220],
      "bystander_centered_in_fov": true,
      "social_reference_sought": true
    }
  ]
}
```

## Verification & Validation Check
- **Singular Video Test**: Pick an ambiguous Ego4D clip where someone builds something and then looks up to show a friend. Print out the `camera_shift_vector` timeline. Assert that the `bystander_centered_in_fov` triggers true concurrently with the frame visually aligning the bystander's face in the center matrix.
- **Batch Test**: During large-scale aggregation, ensure that videos flagged with `social_reference_sought: true` legitimately have a shifting bounding-box origin within the array. Memory constraints are minimal here, but loop efficiency across large coordinate arrays should be benchmarked on the **Mac mini M4 Pro**.

---

## 🧪 Resolved Issues & Implementation Refinements

1. **Inverted Optical Flow Directionality (Resolved - May 04)**:
   - **Problem**: When calculating the camera shift vector, the initial implementation treated the background's optical flow movement directly as the camera's movement. If the camera panned right (+X), the background moved left (-X), resulting in an incorrectly inverted `camera_shift_vector` in the output JSON.
   - **Solution**: Updated the `_extract_camera_shift` method in `pipeline.py` to invert the mean optical flow of the background (`mean_dx = -np.mean(flow[..., 0])`). This accurately estimates the camera's egocentric panning vector based on the static background's relative displacement.

2. **Bystander Centering Logic Scaling (Resolved - May 04)**:
   - **Problem**: The raw bystander bounding boxes lacked dynamic referencing to the source video's resolution, causing the "middle 30%" centering threshold to fail on non-standard Ego4D clip aspect ratios.
   - **Solution**: Refactored the `_check_bystander_centering` function to dynamically extract the `width` and `height` from the `cv2.VideoCapture` properties, establishing a dynamic 35% to 65% bounding box threshold that is resolution-agnostic.

3. **Missing `__init__.py` Module File (Resolved - May 04)**:
   - **Problem**: The `src/layer_03g_shared_reality/` directory was missing an `__init__.py` file, unlike every sibling layer module (e.g., 03f). This caused `ImportError` when attempting to import the pipeline as a Python package rather than via `PYTHONPATH` manipulation.
   - **Solution**: Created `src/layer_03g_shared_reality/__init__.py` with the standard re-export (`from .pipeline import SharedRealityPipeline`), matching the pattern established by `layer_03f_motor_resonance/__init__.py`.

4. **`social_reference_sought` False Positive — No Camera Shift Requirement (Resolved - May 04)**:
   - **Problem**: The `social_reference_sought` flag was set solely based on `bystander_centered_in_fov`, ignoring whether the camera actually panned. According to the specification (Section 3 — "The Social Reference Logic"), social referencing occurs when "the camera explicitly pans away from the task centroid AND centers the bystander." A bystander who was already centered *before* the task climax would produce a false positive, as there was no deliberate gaze shift.
   - **Solution**: Added a `shift_magnitude` check (`np.sqrt(dx² + dy²) > 30.0` pixels) as a conjunction with `bystander_centered_in_fov`. The `social_reference_sought` flag now requires both conditions to be true, matching the documented two-condition social referencing logic.

5. **Missing Bystander Gate in `process_video` (Resolved - May 04)**:
   - **Problem**: Unlike the sibling 03f layer, which gates on `not bystanders or not tasks`, the 03g layer only gated on `not tasks`. Videos with empty `bystander_detections` arrays would still be processed and produce result entries with `bystander_centered_in_fov: false` and `social_reference_sought: false`, polluting the output JSON with meaningless records.
   - **Solution**: Added the missing `not bystanders` check alongside `not tasks` in `process_video()`, causing the pipeline to return `None` early for videos without bystander data.

6. **`np.bool_` Type Leak Into JSON Output (Resolved - May 04)**:
   - **Problem**: The `shift_magnitude > 30.0` comparison returned `np.bool_` (a numpy scalar), not a native Python `bool`. When combined with `bystander_centered` via `and`, the result type was `np.True_` or `np.False_`. This caused `isinstance(value, bool)` checks to fail and would produce non-standard JSON serialization in some environments.
   - **Solution**: Wrapped the expression with `bool(...)` to guarantee a native Python boolean is stored in the output dictionary.

7. **Deferred Gaze Telemetry Integration (Resolved - May 05)**:
   - **Problem**: The specification mentions using raw eye-tracking telemetry (`gaze_x`, `gaze_y` arrays) from Aria glasses datasets as a high-precision fallback for social referencing detection. However, this requires complex `.vrs` parsing and a 200MB dependency, which only benefits a fraction of the dataset.
   - **Solution**: Deferred the implementation of Aria gaze telemetry parsing to a future update to maintain a lighter footprint and rely on optical flow and bounding box centering, which covers all clips uniformly. Added a "Future Possible Direction" section in the Implementation Strategy.

8. **Optical Flow Downsampling Artifacts (Resolved - May 05)**:
   - **Problem**: The Farneback optical flow computation was heavily downsampling frames by a factor of 0.25 to maintain low latency. This aggressive reduction caused subtle micro-pans to fall below the optical flow noise floor, potentially missing important gaze shifts towards bystanders.
   - **Solution**: Updated `_extract_camera_shift` in `pipeline.py` to use a 0.5x downsampling factor. This recovers significant spatial resolution for detecting micro-pans with an acceptable performance trade-off on the Mac mini M4 Pro.

9. **Camera Shift Magnitude Threshold Tuning (Resolved - May 05)**:
   - **Problem**: The `social_reference_sought` logic used an absolute shift magnitude threshold of `30.0` pixels, which was calibrated for 480p clips but was overly permissive and prone to false positives for higher resolution (e.g. 1080p, 4K) footage.
   - **Solution**: Replaced the static absolute threshold in `process_video` with a resolution-normalized threshold dynamically computed as 4% of the frame's diagonal (`threshold = frame_diagonal * 0.04`). This properly scales the sensitivity across varying video resolutions.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Bystander Centering Has No Temporal Ordering Constraint
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:240-249` and `pipeline.py:130-131`. `_check_bystander_centering` returns `True` if *any* bystander bbox centroid falls in the middle 35%–65% region at *any* timestamp inside the reaction window. The downstream `social_reference_sought` then conjuncts this with a global `shift_magnitude > threshold` derived from accumulated optical flow over the *same* window. The two signals are computed independently with no temporal causality check: a video where the bystander was already centered at `start_sec` and the camera panned for an unrelated reason (handing the actor a tool, repositioning the head) would falsely trigger `social_reference_sought = True`. The documented intent — "the camera explicitly pans away from the task centroid AND centers the bystander's bounding box" — explicitly requires a transition (away → toward bystander), not coincidental centering anywhere in the window.

**Option A (recommended)**: **Final-Frame Centering Check** — Replace "any frame in window" with "centering must hold during the last 25% of the reaction window." Pre-centering at task climax is excluded; final stable centering after the camera shift is required.
  - *Pros*: Captures the documented "pan away then center" pattern; minimal logic change (filter timestamps to `t >= start_sec + 0.75 * window_duration`); no schema change.
  - *Cons*: Misses cases where the social reference is brief (glance and back); requires calibrating the 25% tail fraction.

**Option B**: **Centering-Before-vs-After Comparison** — Check centering separately in the first 25% and last 25% of the window. Flag `social_reference_sought` only if the bystander was *not* centered initially but *is* centered at the end.
  - *Pros*: Directly encodes the "transition" semantic; rules out pre-centered-bystander false positives.
  - *Cons*: Requires bystander bbox samples in both temporal segments; videos with sparse bbox sampling may produce ambiguous results.

**Option C**: **Bystander-Aligned Shift Direction** — Compute the vector from the frame center to the bystander centroid at `end_sec`, and require the optical-flow shift vector to be aligned (cosine similarity > 0.6). Drop the binary "centered in middle 30%" gate.
  - *Pros*: Strongest semantic match with "the camera panned toward the bystander"; handles bystanders not exactly in the central rectangle.
  - *Cons*: Larger refactor; needs to handle multi-bystander tiebreaking; affected by the foreground-pollution issue (Issue 4 below).

Your selection: _____

---

### Issue 2: Skipped Videos Reprocessed on Every Resume
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:60-78`. When `process_video` returns `None` (file missing at line 88, no bystanders/tasks at line 95, or every reaction window is malformed at line 104), the `if result:` guard skips both `results.append(...)` and `self.processed_ids.add(video_id)`. Subsequent resume runs re-iterate these videos, re-execute Farneback optical flow over the entire reaction window in `_extract_camera_shift`, re-open `cv2.VideoCapture` three times per task, and re-iterate every bystander bbox in `_check_bystander_centering`, only to discard the result again. Errors caught at line 77 also never mark `video_id` processed.

**Option A (recommended)**: **Sentinel-Record Tracking** — When `process_video` returns `None`, write a sentinel record (e.g., `{"video_id": ..., "layer": "03g_shared_reality", "tasks_analyzed": [], "skipped_reason": "no_bystanders" | "missing_video" | "no_valid_tasks"}`) to results and mark the id processed.
  - *Pros*: Persists skip decisions; downstream consumers gain explicit visibility into skip reasons; resume cost drops to O(JSON-load + set-membership).
  - *Cons*: Output JSON inflates with empty entries; downstream consumers must filter on `tasks_analyzed` length or `skipped_reason`.

**Option B**: **Skip Manifest Sidecar** — Maintain `03g_skipped.json` listing video_ids that produced no output, short-circuit them at the top of the per-entry loop.
  - *Pros*: Keeps the main result JSON clean; explicit skip log is easy to audit.
  - *Cons*: Two-file state machine to maintain; risk of skip-manifest/result-manifest drift if writes aren't atomic.

**Option C**: **Always-Mark-Processed Policy** — Always add `video_id` to `processed_ids` after `process_video` returns, persisting `processed_ids` separately.
  - *Pros*: Cleanly decouples "did we attempt this?" from "did it produce output?".
  - *Cons*: Adds a third state file; loses the rationale for *why* a video was skipped.

Your selection: _____

---

### Issue 3: Three Redundant `cv2.VideoCapture` Opens Per Task
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:108-123`. For each task in each video, the pipeline opens `cv2.VideoCapture(str(video_path))` three times sequentially: once inside `_extract_camera_shift` (line 154), once inside `_check_bystander_centering` (line 217), and once inline at line 117 to fetch frame width/height for the threshold computation. Each open performs codec parsing, container demuxing initialization, and (depending on backend) potentially a full container index scan — costs that scale poorly on long-GOP H.264/H.265 Ego4D clips. The third open at line 117 is particularly wasteful because it only reads two metadata properties already available inside `_check_bystander_centering` (lines 221-222).

**Option A (recommended)**: **Cache Frame Metadata at the Top of `process_video`** — Open the capture once, read width/height/fps, close, and pass these into both helper methods as plain values. `_check_bystander_centering` no longer needs to open a capture at all (it never reads frame data, only metadata). `_extract_camera_shift` still needs its own open for frame iteration.
  - *Pros*: Reduces opens from 3 → 2 per task; eliminates the duplicate metadata fetch; minor refactor; no schema change.
  - *Cons*: Helper signatures must change to accept (width, height); mocked tests need to be updated.

**Option B**: **Reuse a Single Capture Across All Operations Per Task** — Open the capture once at the top of `process_video`, pass it into both helpers (which seek to their respective frame ranges), and release at the end.
  - *Pros*: Maximum reuse; one open per task; aligns with cv2 best practices.
  - *Cons*: Helpers can no longer be unit-tested in isolation with file paths; sharing a stateful capture across helpers is error-prone (seek state); error recovery becomes harder.

**Option C**: **Process-Wide LRU Capture Cache** — Maintain an LRU cache of `cv2.VideoCapture` objects keyed by video_path within the pipeline instance.
  - *Pros*: Fastest for repeated access patterns; transparent to call sites.
  - *Cons*: Capture objects hold OS file descriptors; cache eviction needs explicit `release()` to avoid FD exhaustion; over-engineering for the current 3-open pattern.

Your selection: _____

---

### Issue 4: Background Optical Flow Polluted by Foreground Bystander Motion
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:189-200`. The camera shift estimate computes `mean_dx = -np.mean(flow[..., 0])` and `mean_dy = -np.mean(flow[..., 1])` over the *entire* downsampled frame, including the pixels of the bystander. The inline comment at lines 192-193 acknowledges the assumption — "average flow is an estimation of background movement (assuming background is dominant)" — but provides no enforcement. When the bystander walks across the frame during the reaction window, their motion contaminates `mean_dx`/`mean_dy`, fabricating a phantom "camera shift" magnitude that can push `shift_magnitude` over the resolution-normalized threshold (`frame_diagonal * 0.04`). This produces false-positive `social_reference_sought = True` even when the camera was static. The same issue is independently flagged in 03f Issue 3 (`mean_v` pollution).

**Option A (recommended)**: **Mask Out Bystander Bounding Boxes Before Averaging** — Before computing the mean over `flow[..., 0]`/`flow[..., 1]`, zero out (or mask) the regions inside each bystander bbox at the corresponding timestamp. Compute the mean only over remaining (background) pixels using `np.mean(flow[..., 0][~mask])`.
  - *Pros*: Aligns implementation with the documented "background pixels" intent; cheap mask construction; bystander bboxes are already passed into `process_video`.
  - *Cons*: Requires plumbing bystander bboxes into `_extract_camera_shift`; rectangular masks may exclude legitimate background at the edges; bbox samples are sparse so per-frame mask reconstruction needs interpolation.

**Option B**: **Use the Median Instead of Mean** — Replace `np.mean` with `np.median`. As long as the bystander occupies < 50% of frame pixels, the median is robust to localized foreground motion.
  - *Pros*: One-line change; no bbox plumbing; resistant to localized motion regardless of bbox availability.
  - *Cons*: Fails when foreground exceeds 50% of frame (close-up bystander); biases toward zero in static scenes; loses signal magnitude for small genuine pans.

**Option C**: **Border-Strip Sampling** — Compute `mean_dx`/`mean_dy` only over the outer 20% border of the frame.
  - *Pros*: No bbox plumbing; deterministic background sampling.
  - *Cons*: POV close-quarters scenes (kitchen, table) often have no clear "edge background"; may falsely zero the signal.

Your selection: _____

---

### Issue 5: `_extract_camera_shift` Reads Every Source Frame at Full FPS
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:181-203`. The method seeks to `start_frame = int(start_sec * fps)` and increments `current_frame_idx += 1` in every iteration, reading and Farneback-flowing *every* frame in the reaction window. For a 30 FPS source with a 5-second reaction window, this is 150 dense optical-flow computations per task. The same pattern exists in sister layers 03d (`_extract_ego_motion_noise`) and 03f (`_extract_ego_motion`). The 0.5x spatial downsample (per Resolved Issue 8) helps per-frame cost, but no temporal subsampling occurs.

**Option A (recommended)**: **Adaptive Frame Stride for Optical Flow** — Skip frames so effective flow-FPS is ~10 Hz (`frame_stride = max(1, int(fps / 10))`), iterating `current_frame_idx += frame_stride`. The accumulated shift remains valid because per-frame `mean_dx * 2.0` already represents the displacement between consecutive *processed* frames; with stride > 1, each accumulation step represents a larger inter-frame displacement that should NOT be re-multiplied by stride (the shift is already proportional to elapsed inter-frame time).
  - *Pros*: 3x speedup on 30 FPS videos; pan detection survives because pans last several seconds; no schema change; aligns with proposed 03f optimization.
  - *Cons*: Sub-second jolts may be missed; sister layers should adopt the same stride for consistency; needs validation that accumulated shift remains comparable to current values.

**Option B**: **Use Sparse Feature Tracking (`cv2.calcOpticalFlowPyrLK`)** — Replace dense Farneback with sparse Lucas-Kanade tracking on a grid of corner features (`cv2.goodFeaturesToTrack`). Compute mean shift over surviving features.
  - *Pros*: 10-100x faster than dense Farneback; naturally tracks textured background features more than blank foreground; partially mitigates Issue 4 (foreground pollution) since features are usually on textured background.
  - *Cons*: More moving parts (feature reseeding, tracking failures); requires picking a feature density; less robust on low-texture frames.

**Option C**: **First-vs-Last Frame Subtraction** — Skip optical flow entirely; compute camera shift as ORB/SIFT feature-match displacement between the first and last frame of the window.
  - *Pros*: Constant cost regardless of window length; explicit net displacement (matches the "panned away" semantic).
  - *Cons*: Sensitive to lighting changes between endpoints; misses non-monotonic pans; new dependency on feature matching.

Your selection: _____

---

### Issue 6: Hardcoded Heuristic Constants
**Status**: ⚠️ Confirmed Unresolved — Magic numbers governing classification decisions are inlined as literals: spatial-downsample factor `0.5` at `pipeline.py:174, 187`, upscale factor `2.0` at `pipeline.py:199-200` (must mirror the downsample factor), centering bounds `0.35`/`0.65` at `pipeline.py:228-231`, threshold ratio `0.04` at `pipeline.py:123`, and the unreachable fallback `30.0` at `pipeline.py:125`. None are exposed via constructor arguments, configuration, or class-level constants. Of particular concern: the `0.5` and `2.0` constants are coupled (one is the inverse of the other) but linked only by convention, so an edit to one without the other silently miscalibrates the camera-shift magnitude.

**Option A (recommended)**: **Class-Level Tuning Constants Block** — Hoist constants into a `# --- Detection Tuning ---` block at the top of `SharedRealityPipeline` (e.g., `OPTICAL_FLOW_DOWNSAMPLE = 0.5`, `CENTERING_LOWER_BOUND = 0.35`, `CENTERING_UPPER_BOUND = 0.65`, `SHIFT_THRESHOLD_RATIO = 0.04`). Derive the upscale factor as `1.0 / OPTICAL_FLOW_DOWNSAMPLE` to enforce the inverse coupling; remove the unreachable `30.0` fallback (since the corresponding code path also produces an all-zero shift vector, making the threshold moot).
  - *Pros*: Centralizes tuning surface; eliminates the silent-decoupling bug between downsample/upscale; easy to override via subclassing; zero runtime overhead.
  - *Cons*: Still requires code edit to retune in production; not externally configurable per-batch.

**Option B**: **Config File (YAML/JSON)** — Load constants from a `shared_reality_config.yaml` co-located with the manifest path.
  - *Pros*: Non-developers can retune; supports per-experiment configurations; auditable as artifacts.
  - *Cons*: Adds a config-loader dependency; tests must construct config fixtures; potential for misconfiguration drift.

**Option C**: **Constructor Arguments with Defaults** — Add named parameters to `__init__` with sensible defaults pulled from current literals.
  - *Pros*: Pythonic; tests can override per-test; type-hints document the tuning surface.
  - *Cons*: Constructor-signature growth; orchestration code must thread parameters through.

Your selection: _____
