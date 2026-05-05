# AI Task Breakdown: Attention / Engagement Layer (03a)

## Objective
The **Attention/Engagement Layer** is a specialized social feature layer designed to analyze the cognitive state and focus of the other human(s) present in the scene. Once a video passes the Social Presence Filter, this layer determines whether the external actor is actively paying attention to the POV wearer or interacting with something else.

---

## 📥 Input Requirements
This layer relies directly on the outputs of the previous filtering pipeline:
- **`filtered_manifest.json`** (required): This layer will *only* process clips that encompass genuine social interactions, safely ignoring empty rooms or isolated tasks.
- **`bystander_detections` array** (required): The per-person bounding boxes and timestamps persisted by Node 02's Social Presence Filter. This layer uses these to crop and track each bystander without re-running YOLO.
- **Cross-layer (optional)**: None. This layer has no sibling-layer dependencies.

---

## 🛠️ Implementation Strategy

### 1. Actor Tracking & Isolation
For **each** `person_id` in the manifest's `bystander_detections` array, use the pre-computed bounding boxes as initialization for temporal face/body tracking throughout the clip. If multiple bystanders are present, each is tracked and scored independently. This ensures we capture the attention state of every visible person, not just the most prominent one.

### 2. Frame Sampling Strategy
Running inference on every frame is prohibitively expensive. Use the following tiered sampling approach:
- **Default stride**: Sample 1 frame every **0.5 seconds** (2 FPS effective). This is sufficient for head-pose estimation which changes relatively slowly.
- **Adaptive boost**: If the attention score changes by more than `0.3` between two consecutive samples, temporarily increase to **5 FPS** for the next 2 seconds to capture the transition with finer resolution.
- **Alignment**: Sampling timestamps should be snapped to the nearest frame boundary using the clip's native `fps` from the manifest to avoid inter-frame interpolation artifacts.

### 3. Gaze and Head-Pose Estimation & Target Mapping
We must determine not only if the bystander is focusing on the POV wearer's face (the camera lens) but also if they are paying attention to the *action* being performed (e.g., the POV actor's hands). 
- **SOTA 3D Gaze Raycasting (Recommended for batch)**: 
  1. Use a state-of-the-art 3D gaze estimation model (e.g., **L2CS-Net** or **CrossGaze**) to regress pitch and yaw directly from the bystander's cropped face.
  2. Project this 3D gaze vector into a 2D "ray" across the video frame.
  3. Validate the intersection of this ray against two primary targets: the camera center (POV actor's face/eyes) and the POV actor's hands (detected via an egocentric hand detector like MediaPipe Hands or an Ego4D-trained object detector).
  4. The final `attention_score` is derived from the minimum distance between the projected gaze ray and these target regions. High attention is scored if the bystander is watching *either* the camera lens *or* the task/hands.
- **VLM Approach (Recommended for validation/spot-check)**: Use local Vision Language Models (e.g., Ollama running `moondream` or `Qwen2.5-VL`) on sampled frames to perform cognitive state classification, asking specific prompts like *"Is the person in the frame looking directly at the camera or at the task being performed? Respond with a confidence score from 0 to 100."*

### 4. Attention Scoring & Temporal Trace
For each bystander, compile the per-sample attention values into a **temporal attention trace**—a timeseries of `(timestamp, score)` pairs. From this trace, derive summary statistics:
- `average_attention_score`: Mean across all samples.
- `peak_engagement_timestamp_sec`: Timestamp of the highest single-sample score.
- `attention_variance`: Variance of the trace (high variance = flickering attention).
- `sustained_engagement_sec`: Longest contiguous window where score ≥ 0.7.

The raw trace is critical for downstream layer correlation (e.g., 03b Reasonable Emotion can check if the bystander was even looking when the emotion was detected).

---

## 📤 Output Schema and Integration
In adherence to the Ongoing Layers Paradigm, this layer will *never* modify the original video. It will produce an isolated JSON output, keyed by the `video_id`. When multiple bystanders exist, each gets their own entry in the `per_person` array, plus the file includes an aggregated summary.

**Example Output Data (`03a_attention_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03a_attention",
  "processing_meta": {
    "model_used": "l2cs_net_3d_gaze",
    "sampling_fps_effective": 2.0
  },
  "per_person": [
    {
      "person_id": 0,
      "average_attention_score": 0.85,
      "peak_engagement_timestamp_sec": 4.2,
      "attention_variance": 0.04,
      "sustained_engagement_sec": 6.1,
      "is_engaged": true,
      "gaze_target_classification": "POV_Actor_Hands",
      "attention_trace": [
        {"t": 0.0, "score": 0.72, "pitch_rad": -0.12, "yaw_rad": 0.05},
        {"t": 0.5, "score": 0.81, "pitch_rad": -0.08, "yaw_rad": 0.03},
        {"t": 1.0, "score": 0.90, "pitch_rad": -0.04, "yaw_rad": 0.02},
        {"t": 1.5, "score": 0.88, "pitch_rad": -0.05, "yaw_rad": 0.01}
      ]
    }
  ],
  "aggregate": {
    "num_bystanders_tracked": 1,
    "mean_attention_all_persons": 0.85,
    "any_person_engaged": true
  }
}
```

This dehydrated result can then be successfully merged into the master database for later end-to-end Hugging Face Dataset packaging. The `attention_trace` timeseries is specifically designed to be consumable by sibling layers for temporal correlation.

## Verification & Validation Check
To validate the reliability of the attention scoring mechanics:
- **Singular Video Test**: Process a single known interaction video. Output the `attention_trace` timeseries and write a quick visualization script (e.g., using `matplotlib`) to graph the `attention_score` over time alongside the video timeline. Manually verify if the peaks visually match the moments the bystander looks at the POV camera/hands.
- **Batch Test**: Point the layer script at a batch of 100 clips from the `filtered_manifest.json`. During this batch, actively monitor the process on the **24GB RAM Mac mini M4 Pro** to ensure 3D Gaze Estimation tensor operations run stably via MPS without memory leaks over prolonged loops. Assert that the resulting `03a_attention_result.json` handles missing detections gracefully and outputs valid scores bounded between 0 and 1.

---

## 🚀 Implementation Status

The Attention Layer is fully operational in `src/layer_03a_attention/pipeline.py`. It utilizes 3D Gaze Raycasting (L2CS-Net) to analyze bystander focus relative to the POV camera and task environment, with support for adaptive sampling and automated temporal metric extraction.


## 🧪 Resolved Issues & Implementation Refinements

1. **L2CS Pipeline Crash on Detector Bypass (Resolved - April 28)**:
   - **Problem**: The pipeline passes `include_detector=False` to the L2CS `Pipeline` constructor because bounding boxes are pre-computed by Node 02. However, the upstream `Pipeline.step()` method unconditionally called `np.stack(bboxes)`, `np.stack(landmarks)`, and `np.stack(scores)` on empty lists after the `predict_gaze` call, raising a `ValueError` on every inference.
   - **Solution**: Patched `models/l2cs-net/l2cs/pipeline.py` to guard all three `np.stack()` calls with empty-list checks, returning `np.empty((0, 4))`, `np.empty((0, 5, 2))`, and `np.empty((0,))` for bboxes, landmarks, and scores respectively when no detector is used.

2. **Gaze Vector Convention Mismatch (Resolved - April 28)**:
   - **Problem**: The pipeline's 3D dot-product heuristic in `_track_and_score` used a coordinate system that mismatched L2CS's internal `gazeto3d` utility. Specifically, `v_cam_z` was set to positive Z, while the L2CS coordinate frame uses negative Z to represent "into the screen". This resulted in inverted or nonsensical attention scores.
   - **Solution**: Replaced the hand-rolled vector math in `_track_and_score` with the L2CS `gazeto3d` convention: `v_look = (-cos(yaw)*sin(pitch), -sin(yaw), -cos(yaw)*cos(pitch))`. Corrected `v_cam_z` to `-float(w)` to align with the L2CS coordinate frame.

3. **Dead Code Cleanup in Engagement Logic (Resolved - April 28)**:
   - **Problem**: Leftover implementation stubs and unused variables (`max_sustained`, `current_sustained`) remained in `pipeline.py`, making the production logic difficult to audit.
   - **Solution**: Removed redundant loops and variables, strictly enforcing the validated timestamp-based sustained engagement calculation.

4. **Silent Exception Handling in Inference (Resolved - April 28)**:
   - **Problem**: An `except Exception as e: pass` block caught and discarded all errors from L2CS inference, making systematic model failures (like MPS backend crashes) invisible during batch runs.
   - **Solution**: Implemented explicit error logging to the console while maintaining pipeline continuity, ensuring that hardware or tensor shape errors are surfaced for debugging.

5. **Invalid Indexing in Gaze Inference (Resolved - April 30)**:
   - **Problem**: The pipeline attempted to access angles via `results.pitch[0][0]`, but the L2CS `Pipeline.step()` returns 1D arrays of shape `(N,)`. This raised an `IndexError` on every frame.
   - **Solution**: Updated `pipeline.py` to use 1D indexing (`results.pitch[0]` and `results.yaw[0]`).

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Inaccurate Geometric Heuristic (Camera Intrinsics)
**Status**: ⚠️ Confirmed Unresolved — The dot-product algorithm in `_track_and_score` (pipeline.py, line ~269) assumes a simplified pinhole camera model with a hardcoded focal length approximation (`-float(w)` for `v_cam_z`). Ego4D videos are captured with various camera rigs (including Aria glasses with fisheye lenses), where the actual focal length and distortion parameters differ significantly from this assumption. This can cause systematic over- or under-estimation of the gaze-to-camera angle, especially at frame edges.

**Option A (recommended)**: **Ego4D Camera Intrinsics Lookup** — Parse the `ego4d.json` metadata for per-clip camera parameters (`focal_length`, `principal_point`, `distortion_coefficients`) when available. Use these to compute a calibrated `v_cam_z` value. Fall back to the current heuristic when metadata is absent.
  - *Pros*: Significantly improves gaze accuracy for clips with known intrinsics; non-breaking (fallback preserves current behavior); Ego4D provides this data for Aria glasses clips.
  - *Cons*: Not all Ego4D clips have camera intrinsics; requires parsing additional metadata fields; different camera models may need different distortion correction.

**Option B**: **Learned Gaze Calibration Offset** — Run a calibration pass on a small labeled subset (e.g., 20 clips with manually verified attention scores) to learn a per-camera-type correction factor. Apply this as a multiplicative offset to the raw dot-product score.
  - *Pros*: Works even without explicit intrinsics; captures systematic biases across camera types.
  - *Cons*: Requires manually labeled calibration data; may overfit to the calibration set; ongoing maintenance as new camera types are added.

Your selection: Proceed with Option A.

---

### Issue 2: Missing Actor Hand Detection
**Status**: ⚠️ Confirmed Unresolved — The current implementation (pipeline.py) only scores attention towards the camera lens (centroid of frame). The specification in Section 3 explicitly states: "High attention is scored if the bystander is watching *either* the camera lens *or* the task/hands." Hand bounding boxes are not available in the `filtered_manifest.json` schema and are not computed by any upstream module.

**Option A (recommended)**: **Integrate MediaPipe Egocentric Hand Detection in Node 02** — Add a `hand_detections` field to the `filtered_manifest.json` schema (additive, non-breaking). Run MediaPipe Hands during the social presence filter pass to detect and persist wearer hand bounding boxes. Layer 03a then checks gaze intersection against both the camera centroid and the hand regions.
  - *Pros*: MediaPipe is already listed in `ml_dependencies.md`; runs on CPU with minimal overhead (~5ms/frame); provides hand landmark coordinates for future layers.
  - *Cons*: Adds ~5% processing time to Node 02; schema change requires updating all downstream consumers; MediaPipe hand detection is less reliable on heavily occluded egocentric views.

**Option B**: **Heuristic Hand Zone Assumption** — Assume the wearer's hands are always in the bottom-center third of the frame (a reasonable assumption for most egocentric tasks like cooking, crafting, etc.). Score attention towards this zone in addition to the camera centroid.
  - *Pros*: Zero additional dependencies or models; trivial to implement; no schema changes needed.
  - *Cons*: Inaccurate when hands are raised (e.g., gesturing, pointing); does not adapt to the actual hand position; will produce false positives for bystanders looking at the ground.

Your selection: Proceed with Option A.

---

### Issue 3: Tracking Loss (Person ID Consistency)
**Status**: ⚠️ Confirmed Unresolved — Node 02's `bystander_detections` assigns `person_id` based on detection order per frame, not via temporal tracking. When multiple bystanders overlap or cross paths between sampled frames, the same physical person can receive different `person_id` values. This causes Layer 03a to produce fragmented attention traces where a single bystander's gaze data is split across multiple `person_id` entries.

**Option A (recommended)**: **ByteTrack Integration in Node 02** — Replace the per-frame ID assignment with [ByteTrack](https://github.com/ifzhang/ByteTrack), a lightweight, high-performance multi-object tracker. ByteTrack assigns temporally consistent IDs using Kalman filtering and IoU-based association. It runs on CPU with negligible overhead.
  - *Pros*: SOTA tracking accuracy; minimal compute cost (~1ms/frame); maintains consistent person_id across occlusions; widely adopted in detection pipelines.
  - *Cons*: Adds a dependency (`byte_track` or inline implementation); requires refactoring the `bystander_detections` writer in Node 02; existing test manifests may need regeneration.

**Option B**: **IoU-Based Greedy Matching** — For each new frame, match detected persons to the previous frame's persons using IoU of bounding boxes. Assign the same `person_id` to the highest-IoU match; create new IDs only for unmatched detections.
  - *Pros*: Simple to implement (~30 lines of code); no external dependencies; works well for non-overlapping, slow-moving subjects.
  - *Cons*: Fails catastrophically during occlusions and fast crossings; no motion model to predict through gaps; inferior to dedicated trackers.

Your selection: Proceed with Option A.

---

### Issue 4: Inefficient Temporal Seeking
**Status**: ⚠️ Confirmed Unresolved — The pipeline uses `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)` for adaptive sampling, which on macOS requires seeking to the nearest keyframe and then decoding forward. For non-keyframe-aligned timestamps, this results in redundant decode cycles that can be 10-50x slower than sequential reads.

**Option A (recommended)**: **Sequential Read with Frame Skip** — Replace random-access seeking with a single forward pass. Read frames sequentially from `start_sec` to `end_sec`, processing only frames whose index matches the sampling schedule (`frame_idx % stride == 0`). Skip non-target frames with `cap.grab()` (fast discard without decoding).
  - *Pros*: Eliminates all seeking overhead; `cap.grab()` is ~10x faster than `cap.read()` for skipped frames; works identically across all codecs and OS.
  - *Cons*: Must process the entire reaction window linearly (cannot skip to arbitrary timestamps mid-window); slightly more complex loop logic.

Your selection: Proceed with Option A.

---

### Issue 5: VLM Bottleneck in E2E Pipeline
**Status**: ⚠️ Confirmed Unresolved — Node 02's task labeling uses Qwen2.5-VL for climax refinement on slow/cognitive tasks. During full E2E pipeline runs (Node 02 → 03a → 03b → ...), the VLM inference in Node 02 creates a latency bottleneck that delays all downstream layer execution. For geometry-only verification runs (e.g., testing 03d Proxemic Kinematics), the VLM step is unnecessary overhead.

**Option A (recommended)**: **`--skip-vlm` CLI Flag** — Add a command-line flag to the Node 02 pipeline that skips the Stage 2 VLM refinement step entirely. When set, the pipeline uses only the optical flow peak as the `task_climax_sec` and marks `climax_extraction_method` as `"optical_flow_peak_only"`. This allows fast geometry-only runs without loading the 3-10GB VLM.
  - *Pros*: Trivial to implement (single `if` guard around the VLM call); no change to output schema (method field updates naturally); user-controlled.
  - *Cons*: Reduces climax accuracy for slow tasks when used; users must remember to re-run with VLM for final production exports.

**Option B**: **Pre-Computed Manifest Cache** — After a full VLM-enhanced run, cache the final `filtered_manifest.json` as a "gold" manifest. Subsequent E2E runs check for this cached manifest first, skipping Node 02 entirely when the video set hasn't changed.
  - *Pros*: Zero latency for downstream layers on repeat runs; no accuracy compromise.
  - *Cons*: Cache invalidation is fragile (any new video or re-filter requires a full re-run); adds cache management complexity.

Your selection: Proceed with Option A
