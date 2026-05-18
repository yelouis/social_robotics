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
- **Default (baseline) stride**: Sample 1 frame every **0.125 seconds** (8 FPS effective). The 8 FPS floor exists to satisfy Layer 03e's Nyquist requirement: communicative head nods occur up to 3 Hz, and 8 FPS provides a 4 Hz Nyquist ceiling. Lower rates were validated to physically erase fast nods (see 03e Resolved Issue #5).
- **Adaptive boost**: If the attention score changes by more than `0.3` between two consecutive samples, temporarily increase to **16 FPS** (0.0625 s stride) for the next **2 seconds** to capture the transition with finer resolution. After the burst window expires, the stride decays back to the 8 FPS baseline.
- **Alignment**: Sampling timestamps are snapped to the nearest frame boundary using the clip's native `fps` from the manifest to avoid inter-frame interpolation artifacts. Downstream consumers MUST treat `attention_trace[i].t` as authoritative — the stride is not uniform, and any layer that depends on uniform spacing (e.g., Butterworth filters in 03e) must resample the trace onto a fixed-dt grid first.

### 3. Gaze and Head-Pose Estimation & Target Mapping
We must determine not only if the bystander is focusing on the POV wearer's face (the camera lens) but also if they are paying attention to the *action* being performed (e.g., the POV actor's hands). 
- **SOTA 3D Gaze Raycasting (Recommended for batch)**: 
  1. Use a state-of-the-art 3D gaze estimation model (e.g., **L2CS-Net** or **CrossGaze**) to regress pitch and yaw directly from the bystander's cropped face.
  2. Project this 3D gaze vector into a 2D "ray" across the video frame.
  3. Validate the intersection of this ray against two primary targets: the camera center (POV actor's face/eyes) and the POV actor's hands (detected via an egocentric hand detector like MediaPipe Hands or an Ego4D-trained object detector). For accurate projection, extract `focal_length` and `principal_point` from dataset metadata (e.g., `ego4d.json`) to compute a calibrated camera vector. If metadata is missing, it safely falls back to a standard pinhole camera heuristic (`v_cam_z = -width`).
  4. The final `attention_score` is derived from the minimum distance between the projected gaze ray and these target regions. High attention is scored if the bystander is watching *either* the camera lens *or* the task/hands.
- **VLM Approach (Recommended for validation/spot-check)**: Use local Vision Language Models (e.g., Ollama running `moondream` or `Qwen2.5-VL`) on sampled frames to perform cognitive state classification, asking specific prompts like *"Is the person in the frame looking directly at the camera or at the task being performed? Respond with a confidence score from 0 to 100."*

### 4. Attention Scoring & Temporal Trace
For each bystander, compile the per-sample attention values into a **temporal attention trace**—a timeseries of `(timestamp, score)` pairs. From this trace, derive summary statistics:
- `average_attention_score`: Mean across all samples.
- `peak_engagement_timestamp_sec`: Timestamp of the highest single-sample score.
- `attention_variance`: Variance of the trace (high variance = flickering attention).
- `sustained_engagement_sec`: Longest contiguous window where score ≥ 0.7.

The raw trace is critical for downstream layer correlation (e.g., 03b Reasonable Emotion can check if the bystander was even looking when the emotion was detected).

### 5. Memory Management & Orchestration
Because the Layer 03 processing suite contains multiple VLM and vision models, memory orchestration is critical. The Attention Layer provides a context-manager based `unload()` method to free the L2CS-Net ResNet50 graph and clear the MPS cache after inference. 
On high-memory hosts (≥ 48 GB unified memory, such as the Mac Studio M4 Max), the full suite of Layer 03 models can remain resident simultaneously. The pipeline dynamically detects the host's memory capacity and skips the unload sequence if sufficient memory is available, eliminating the latency penalty of reloading L2CS for subsequent videos.

### 6. Model Selection Rationale
The pipeline uses **L2CS-Net** (ResNet50 backbone) for 3D Gaze Estimation. While newer models like CrossGaze and 3DGazeNet offer marginally better angular accuracy (e.g., ~7° vs ~10° on Gaze360), L2CS-Net's ~200 MB footprint fits comfortably within the memory constraints of smaller 24 GB hosts. Since downstream layers have not demonstrated any failures attributable to L2CS pitch/yaw noise, the project defers upgrading to heavier models to avoid regression risk and maintain broad hardware compatibility.

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
    "sampling_fps_effective": 8.0,
    "sampling_fps_burst": 16.0,
    "sampling_strategy": "adaptive_8_to_16_fps"
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
        {"t": 0.0,    "score": 0.72, "pitch_rad": -0.12, "yaw_rad": 0.05, "target": "Camera"},
        {"t": 0.125,  "score": 0.81, "pitch_rad": -0.08, "yaw_rad": 0.03, "target": "Camera"},
        {"t": 0.1875, "score": 0.90, "pitch_rad": -0.04, "yaw_rad": 0.02, "target": "POV_Actor_Hands"},
        {"t": 0.25,   "score": 0.88, "pitch_rad": -0.05, "yaw_rad": 0.01, "target": "POV_Actor_Hands"}
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

> **Per-sample `target`**: Each trace point now carries a `target` label — one of `"Camera"`, `"POV_Actor_Hands"`, or `"Unknown"` — indicating which raycast geometry produced the maximum dot-product for that sample. The per-person `gaze_target_classification` is the majority `target` across the trace (excluding `"Unknown"`). Downstream layers needing finer-grained per-sample target attribution should consume `attention_trace[i].target` directly rather than relying on the per-person aggregate.

## Verification & Validation Check
To validate the reliability of the attention scoring mechanics:
- **Singular Video Test**: Process a single known interaction video. Output the `attention_trace` timeseries and write a quick visualization script (e.g., using `matplotlib`) to graph the `attention_score` over time alongside the video timeline. Manually verify if the peaks visually match the moments the bystander looks at the POV camera/hands.
- **Batch Test**: Point the layer script at a batch of 100 clips from the `filtered_manifest.json`. During this batch, actively monitor the process on the **Mac Studio (M4 Max, 64 GB unified memory)** to ensure 3D Gaze Estimation tensor operations run stably via MPS without memory leaks over prolonged loops. Assert that the resulting `03a_attention_result.json` handles missing detections gracefully and outputs valid scores bounded between 0 and 1.

---

## 🚀 Implementation Status

The Attention Layer is fully operational in `src/layer_03a_attention/pipeline.py`. It utilizes 3D Gaze Raycasting (L2CS-Net) to analyze bystander focus relative to the POV camera and task environment, with support for adaptive sampling and automated temporal metric extraction.


## 🧪 Resolved Issues & Implementation Refinements

_All previously tracked issues have been successfully integrated into the main architecture documentation or remediated in code._

## ⚠️ Unresolved Issues & Suggestions

_None at this time — all tracked issues have transitioned to the Resolved section above._

