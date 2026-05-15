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
- **Batch Test**: During large-scale aggregation, ensure that videos flagged with `social_reference_sought: true` legitimately have a shifting bounding-box origin within the array. Memory constraints are minimal here, but loop efficiency across large coordinate arrays should be benchmarked on the **Mac Studio (M4 Max, 64 GB unified memory)**.

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

10. **Heuristic Constants Hoisted to Class-Level Tuning Block (Resolved - May 08)**:
    - **Problem**: Magic numbers governing classification decisions were inlined as literals throughout the detection path: spatial-downsample factor `0.5` at the Farneback call sites, upscale factor `2.0` (the inverse of the downsample, linked only by convention), centering bounds `0.35`/`0.65`, threshold ratio `0.04`, and the unreachable `30.0` fallback. The downsample/upscale coupling was particularly fragile — editing one literal without the other would silently miscalibrate the camera-shift magnitude.
    - **Solution**: Hoisted all heuristic constants into a `# --- Detection Tuning ---` block at the top of `SharedRealityPipeline`: `OPTICAL_FLOW_DOWNSAMPLE`, `TARGET_FLOW_FPS`, `CENTERING_LOWER_BOUND`, `CENTERING_UPPER_BOUND`, `SHIFT_THRESHOLD_RATIO`, and `FINAL_CENTERING_TAIL_FRACTION`. The upscale factor is exposed as a derived `OPTICAL_FLOW_UPSCALE` property (`1.0 / OPTICAL_FLOW_DOWNSAMPLE`), enforcing the inverse coupling at the source level. The unreachable `30.0` fallback in `process_video` was deleted because the new flow returns sentinels when the metadata read fails (Resolved Issue 18), so the threshold-fallback branch is no longer reachable.

11. **Bystander Centering Has No Temporal Ordering Constraint (Resolved - May 08)**:
    - **Problem**: `_check_bystander_centering` returned `True` if any bystander bbox centroid landed in the middle 35–65% region at *any* timestamp inside the reaction window. The downstream `social_reference_sought` then conjuncted this with a global shift-magnitude check derived from accumulated optical flow over the same window. The two signals were computed independently with no temporal causality check: a video where the bystander was already centered at `start_sec` and the camera panned for an unrelated reason (handing the actor a tool, repositioning the head) would falsely trigger `social_reference_sought = True`. The documented intent — "the camera explicitly pans away from the task centroid AND centers the bystander's bounding box" — explicitly requires a transition (away → toward bystander), not coincidental centering anywhere in the window.
    - **Solution**: `_check_bystander_centering` now restricts its temporal filter to the *final `FINAL_CENTERING_TAIL_FRACTION` (25%) of the reaction window*, computed as `tail_start = end_sec - (end_sec - start_sec) * 0.25`. Pre-centering at task climax is excluded; only stable centering after the camera has had time to pan registers as a hit. Combined with the existing shift-magnitude conjunction in `social_reference_sought`, the layer now requires both a meaningful camera shift *and* end-of-window bystander centering to flag a social-referencing event. The corresponding test cases in `tests/test_layer_03g.py` were updated to use timestamps in the tail of the reaction window, and a new sub-assertion verifies that pre-tail centering does not trigger the flag.

12. **Skipped Videos Reprocessed on Every Resume (Resolved - May 08)**:
    - **Problem**: When `process_video` returned `None` (file missing, no bystanders/tasks, or every reaction window malformed), the `if result:` guard in `run` skipped both `results.append(...)` and `self.processed_ids.add(video_id)`. Subsequent resume runs re-iterated these videos, re-executed Farneback optical flow, re-opened `cv2.VideoCapture` per task, and re-iterated every bystander bbox, only to discard the result again. Errors caught at the outer except similarly never marked `video_id` processed.
    - **Solution**: Introduced a `_sentinel(video_id, reason)` helper that returns `{"video_id": ..., "layer": "03g_shared_reality", "tasks_analyzed": [], "skipped_reason": "missing_video" | "no_bystanders_or_tasks" | "no_valid_tasks"}`, and replaced the four `return None` sites in `process_video` with calls to this helper. The truthiness gate in `run` was removed so every video — including skipped ones — is appended to `results`, written atomically, and added to `self.processed_ids`. Resume cost for previously-skipped videos drops to set-membership lookup, and downstream consumers gain explicit visibility into why a video was excluded. The exception path retains its existing behavior (errors logged but not marked processed) so transient failures remain retryable on rerun. Consumers must filter on `tasks_analyzed` length or the presence of `skipped_reason` to avoid treating sentinel entries as detection results.

13. **Three Redundant `cv2.VideoCapture` Opens Per Task (Resolved - May 08)**:
    - **Problem**: For each task in each video, the pipeline opened `cv2.VideoCapture(str(video_path))` three times sequentially: once inside `_extract_camera_shift`, once inside `_check_bystander_centering`, and once inline in `process_video` to fetch frame width/height for the threshold computation. Each open performed codec parsing, container demuxing initialization, and (depending on backend) potentially a full container index scan — costs that scaled poorly on long-GOP H.264/H.265 Ego4D clips. The third open was particularly wasteful because it only read two metadata properties already available inside `_check_bystander_centering`.
    - **Solution**: `process_video` now opens the capture once at the top, reads `width`/`height`/`fps`, releases the capture, and computes the resolution-normalized threshold from the cached metadata. `_check_bystander_centering`'s signature was changed to `(bystanders, start_sec, end_sec, width, height)` — it no longer opens any capture since it never reads frame data, only metadata. `_extract_camera_shift`'s signature was changed to `(video_path, start_sec, end_sec, width, height, fps, bystanders)` and uses the passed `fps` directly, keeping a single capture open only for frame iteration. Net: 3 opens per task → 2 opens per task with no behavior change. The capture-reading test fixtures in `tests/test_layer_03g.py` were updated to pass `width`, `height`, `fps`, and an empty bystanders list explicitly.

14. **Background Optical Flow Polluted by Foreground Bystander Motion (Resolved - May 08)**:
    - **Problem**: The camera shift estimate computed `mean_dx = -np.mean(flow[..., 0])` and `mean_dy = -np.mean(flow[..., 1])` over the *entire* downsampled frame, including the pixels of the bystander. The inline comment acknowledged the assumption — "average flow is an estimation of background movement (assuming background is dominant)" — but provided no enforcement. When the bystander walked across the frame during the reaction window, their motion contaminated `mean_dx`/`mean_dy`, fabricating a phantom "camera shift" magnitude that could push `shift_magnitude` over the resolution-normalized threshold. The same failure mode is independently flagged in 03f Resolved Issue 11.
    - **Solution**: Added a `_bystander_mask_for_frame(t, frame_shape_ds, bystanders)` helper that builds a boolean mask over the downsampled frame zeroing out the nearest-timestamp bbox for every bystander (rectangles scaled by `OPTICAL_FLOW_DOWNSAMPLE`). `_extract_camera_shift` now accepts the `bystanders` list (plumbed in from `process_video`) and computes `mean_dx`/`mean_dy` over background pixels only via `np.mean(flow[..., k][mask])`, falling back to the unmasked mean only when every pixel is inside a bbox (degenerate case).

15. **`_extract_camera_shift` Read Every Source Frame at Full FPS (Resolved - May 08)**:
    - **Problem**: `_extract_camera_shift` seeked to `start_frame = int(start_sec * fps)` and incremented `current_frame_idx += 1` each iteration, computing dense Farneback flow on *every* frame in the reaction window. For a 30 FPS source and a 5-second window this was 150 dense optical-flow computations per task. The 0.5× spatial downsample helped per-frame cost but no temporal subsampling occurred.
    - **Solution**: Added a `TARGET_FLOW_FPS = 10.0` class constant; `_extract_camera_shift` now computes `frame_stride = max(1, round(fps / TARGET_FLOW_FPS))` and seeks/reads frames in stride-sized hops. Pan detection survives because pans last several seconds (much longer than the new 100 ms inter-sample interval). Calibration: per-iteration flow magnitude scales linearly with stride for continuous motion, and the loop runs `1/stride` as many iterations, so the accumulated `total_dx`/`total_dy` for continuous motion remains the same as the per-frame loop — no extra multiplier needed (matching Option A's note). At 30 FPS this is a 3× speedup with no behavior change on the existing pan-detection test fixture.

16. **Aria Gaze Telemetry Deferral Made Permanent (Resolved - May 15)**:
    - **Problem**: Resolved Issue #7 deferred the Aria glasses `gaze_x`/`gaze_y` telemetry integration when the host was the 24 GB Mac mini M4 Pro, citing the ~200 MB `projectaria_tools` dependency and `.vrs` parser-side memory pressure. On the **Mac Studio (M4 Max, 64 GB unified memory)** target host the dependency is now memory-affordable, raising the question of whether the deferral should be reversed for the Aria subset of Ego4D (where gaze provides a sub-degree ground-truth precision floor that the optical-flow + bounding-box heuristic cannot reach). The decision needed an explicit final resolution rather than another open footnote.
    - **Solution**: Confirmed the deferral as **permanent**. The Aria-glasses subset of the filtered Ego4D corpus is too small (in this project's dataset) to justify the ~200 MB venv-size hit, the `.vrs` per-frame parsing cost, and the two-signal aggregation work in `04_dehydrated_export.md` that an `aria_vrs` `gaze_source` field would require. The "Future Possible Direction: Aria Glasses Gaze Telemetry Integration" subsection in the Implementation Strategy is retained as a documented re-entry point if the Aria-clip share of future ingestion runs grows materially. No code change; the existing optical-flow + bounding-box-centering path remains the single uniform signal.

17. **Optical-Flow Downsample Tiered on Host Memory (Resolved - May 15)**:
    - **Problem**: Resolved Issue #8 pinned `OPTICAL_FLOW_DOWNSAMPLE = 0.5` to fit the legacy 24 GB Mac mini M4 Pro alongside the rest of the 03 layers. On the **Mac Studio (M4 Max, 64 GB unified memory)** target host, with `TARGET_FLOW_FPS = 10.0` (Resolved Issue #15) capping temporal cost and 1080p Ego4D input keeping the per-frame Farneback transient near ~120 MB, the 0.5× downsample was now discarding micro-pan spatial detail that the larger host had headroom to retain. A static class constant could not express both budgets simultaneously.
    - **Solution**: Replaced the single `OPTICAL_FLOW_DOWNSAMPLE = 0.5` class constant with a tiered pair — `OPTICAL_FLOW_DOWNSAMPLE_HIGH_MEM = 1.0`, `OPTICAL_FLOW_DOWNSAMPLE_LOW_MEM = 0.5`, and `HIGH_MEMORY_HOST_BYTES = 48 * 2**30` — plus a `_host_high_memory()` staticmethod that probes `psutil.virtual_memory().total` (falling back to `False` when `psutil` is missing, mirroring `AttentionLayerPipeline.host_can_retain_resident` in 03a and `AcousticProsodyPipeline.host_can_eager_load_sensevoice` in 03c). `SharedRealityPipeline.__init__` resolves the tier once and assigns the chosen value to `self.OPTICAL_FLOW_DOWNSAMPLE`; the `OPTICAL_FLOW_UPSCALE` `@property` reads that instance attribute, so the inverse coupling enforced by Resolved Issue #10 holds at either tier with no other constant changes (the bystander mask helper in Resolved Issue #14 is already parameterized by `OPTICAL_FLOW_DOWNSAMPLE` and re-scales automatically). The Mac Studio path now runs Farneback at full source resolution, recovering sub-degree micro-pan detail; the Mac mini fallback path is byte-for-byte identical to the pre-change behavior. The synthetic `_create_synthetic_video` fixture in `tests/test_layer_03g.py` was switched from a repeating 40 px checkerboard to a seeded random-noise texture so the pan-detection test no longer fails the aperture problem that Farneback exhibits on repetitive patterns at full resolution. The resolution-normalized `SHIFT_THRESHOLD_RATIO = 0.04` was left untouched; if production Ego4D footage at 1.0× shows a meaningfully different noise floor than at 0.5×, a follow-up calibration entry will tune it then.


## ⚠️ Unresolved Issues & Suggestions

*(No unresolved issues at this time.)*
