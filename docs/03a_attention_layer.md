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

6. **Inaccurate Geometric Heuristic (Camera Intrinsics) (Resolved - May 05)**:
   - **Problem**: The dot-product algorithm in `_track_and_score` assumed a simplified pinhole camera model with a hardcoded focal length approximation, causing systematic gaze estimation errors for Aria glasses and other non-standard lenses in Ego4D clips.
   - **Solution**: Updated `AttentionLayerPipeline` to parse `ego4d.json` metadata via `EGO4D_METADATA_PATH` to extract exact `focal_length` and `principal_point` parameters for calibrated `v_cam_z` computation, with fallback to the standard heuristic.

7. **Missing Actor Hand Detection (Resolved - May 05)**:
   - **Problem**: Gaze intersection scoring only considered the camera lens (centroid), neglecting the POV actor's hands. The specification required high attention scoring if the bystander was watching either the camera lens or the task/hands.
   - **Solution**: Updated Node 02's `social_presence.py` to export MediaPipe hand bounding boxes, appended them to `filtered_manifest.json` as `hand_detections`, and updated Layer 03a to compute dot-product intersections against both the camera and all detected hand centroids.

8. **Tracking Loss (Person ID Consistency) (Resolved - May 05)**:
   - **Problem**: Node 02 assigned `person_id` based purely on detection array indices per frame, causing fragmented attention traces when multiple bystanders overlapped or crossed paths across sampled frames.
   - **Solution**: Replaced the per-frame ID assignment in Node 02 with Ultralytics' built-in ByteTrack (`model.track()` with `bytetrack.yaml` and `persist=True`) to assign temporally consistent IDs via Kalman filtering and IoU-based association.

9. **Inefficient Temporal Seeking (Resolved - May 05)**:
   - **Problem**: The pipeline used `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)` for adaptive sampling, resulting in redundant decode cycles and 10-50x slower seeking on macOS due to keyframe decoding overhead for non-keyframe-aligned timestamps.
   - **Solution**: Rewrote the frame fetching logic in `_track_and_score` to use sequential read with frame skipping. We now iterate sequentially using `cap.grab()` to quickly discard non-target frames, completely eliminating random-access seeking overhead.

10. **VLM Bottleneck in E2E Pipeline (Resolved - May 05)**:
    - **Problem**: Node 02's mandatory task labeling used Qwen2.5-VL for climax refinement, creating a massive latency bottleneck that delayed downstream execution for geometry-only verification runs.
    - **Solution**: Added a `--skip-vlm` CLI flag to `run_verification.py` and propagated it to `FilteringPipeline`. When set, it skips the Stage 2 VLM refinement step entirely, relies purely on the optical flow peak, and updates `climax_extraction_method` to `"optical_flow_peak_only"`.

## ⚠️ Unresolved Issues & Suggestions

*None at this time.*


