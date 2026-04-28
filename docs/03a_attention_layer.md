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

## 🚀 Implementation Status & Known Limitations

**What was accomplished:**
The layer has been fully implemented in `src/layer_03a_attention/pipeline.py` using the **SOTA 3D Gaze Raycasting method** (L2CS-Net). It successfully reads `filtered_manifest.json`, leverages the pre-computed bounding boxes from the `bystander_detections` array to crop images of the bystanders, and queries the L2CS ResNet50 model to extract exact `pitch_rad` and `yaw_rad` vectors. 

To determine the final `attention_score`, the layer utilizes a **3D geometric dot-product heuristic** that projects a 3D unit vector representing the bystander's gaze against a unit vector pointing from their bounding box center towards the center of the camera. The layer features adaptive sampling, computes temporal metrics (peak engagement, variance, sustained engagement), and performs safe atomic writes.

**Potential Errors & Solutions:**
1. **Inaccurate Geometric Heuristic**: The dot-product algorithm assumes the focal length is roughly equivalent to the frame width and projects the camera at `(W/2, H/2)`. If the camera is heavily distorted (e.g. fish-eye POV lenses without un-distortion), the projection angle might mismatch the real-world gaze ray.
   - *Solution*: Extract camera intrinsics from the specific dataset (Ego4D or EPIC-KITCHENS) to properly calibrate the `v_cam_z` (focal length) parameter in the projection math.
2. **Missing Actor Hand Detection**: The current heuristic only scores attention towards the POV wearer's *camera/face*. The document requests checking for attention towards the POV actor's *hands* as well.
   - *Solution (Documented)*: Implement a MediaPipe Egocentric Hand Detector in Node 02, log the hand bounding boxes, and pass them into Layer 03a to calculate a secondary dot-product vector `V_hands`.
3. **Tracking Loss**: Because Node 02's output doesn't natively include a robust SORT tracker, bounding boxes might shift IDs across frames if multiple people intersect.
   - *Solution (Documented)*: Implement an explicit DeepSORT or ByteTrack step in Node 02, or build a lightweight bounding-box IoU linker in this layer to strictly maintain `person_id` temporal consistency.
4. **`include_detector=False` Crashes L2CS Pipeline (Resolved)**:
   - **Problem**: The pipeline passes `include_detector=False` to the L2CS `Pipeline` constructor because bounding boxes are pre-computed by Node 02. However, the upstream `Pipeline.step()` method unconditionally called `np.stack(bboxes)`, `np.stack(landmarks)`, and `np.stack(scores)` on empty lists after the `predict_gaze` call, raising a `ValueError` on every inference. This was a critical runtime crash that prevented any real gaze inference from completing.
   - **Solution**: Patched `models/l2cs-net/l2cs/pipeline.py` to guard all three `np.stack()` calls with empty-list checks, returning `np.empty((0, 4))`, `np.empty((0, 5, 2))`, and `np.empty((0,))` for bboxes, landmarks, and scores respectively when no detector is used.

5. **Gaze Vector Convention Mismatch (Resolved)**:
   - **Problem**: The pipeline's 3D dot-product heuristic in `_track_and_score` used `(sin(yaw), -sin(pitch), cos(yaw)*cos(pitch))` to construct the gaze direction vector. However, L2CS's own `gazeto3d` utility defines the convention as `(-cos(yaw)*sin(pitch), -sin(yaw), -cos(yaw)*cos(pitch))`. These are different coordinate systems, meaning the pipeline was silently producing incorrect attention scores. Additionally, the camera-to-bystander direction vector `v_cam_z` was set to `+float(w)` (positive Z), while the L2CS coordinate frame uses negative Z to represent "into the screen" (toward the camera). Together, these two mismatches could invert the meaning of high vs. low attention.
   - **Solution**: Replaced the hand-rolled vector math in `_track_and_score` with the L2CS `gazeto3d` convention: `v_look = (-cos(yaw)*sin(pitch), -sin(yaw), -cos(yaw)*cos(pitch))`. Corrected `v_cam_z` from `+float(w)` to `-float(w)` to align with the L2CS coordinate frame. The test suite confirms that `pitch=0, yaw=0` (looking straight at camera) still produces the expected attention score of `0.92`.

6. **Dead Code in Sustained Engagement Calculation (Resolved)**:
   - **Problem**: Lines 159-165 of `pipeline.py` contained a `for score in scores` loop with a `pass` body and two unused variables (`max_sustained`, `current_sustained`), left over from an incomplete initial implementation. The correct timestamp-based sustained engagement logic was already implemented immediately after this block, making the dead code misleading.
   - **Solution**: Removed the dead loop and unused variables from `pipeline.py`, keeping only the accurate timestamp-based sustained engagement calculation.

7. **Silently Swallowed Gaze Inference Exceptions (Resolved)**:
   - **Problem**: The `except Exception as e: pass` block in `_track_and_score` caught and discarded all errors from L2CS inference, including model crashes, tensor shape mismatches, and MPS backend failures. When inference failed, the frame received a default score of `0.0` with no indication that an error occurred, making systematic model failures invisible during batch runs.
   - **Solution**: Replaced with `print(f"[03a] Gaze inference failed at t={current_t:.2f}s: {e}")` to surface failures in the console log while still allowing the pipeline to continue processing remaining frames.
