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

1. **Redundant Video Decoding for Multiple Bystanders (Resolved - May 23)**:
   - **Problem**: In multi-bystander videos, the Attention pipeline looped over each bystander sequentially. This caused the video file to be opened and decoded from frame 0 multiple times (once per bystander), introducing massive video I/O bottlenecks and preventing parallel tensor operations.
   - **Solution**: Refactored `process_video` and replaced `_track_and_score` with `_track_and_score_batched` in [pipeline.py](file:///Users/louisye/Desktop/Louis/social_robotics/src/layer_03a_attention/pipeline.py) to perform single-pass video decoding. At each sampling timestamp (reconciled as the union of active bystanders' strides), the frame is decoded once, cropped and resized to `(448, 448)` for all active bystanders, stacked into a single `[N, 448, 448, 3]` batch, and processed in a single forward pass of L2CS-Net. This significantly speeds up I/O and leverages Apple Silicon (MPS) batched tensor acceleration.

2. **Human-Face Validity Gate Before Gaze Regression (Resolved - June 03)**:
   - **Problem**: `_track_and_score_batched` cropped every Node-02 bystander bbox and ran L2CS-Net on it unconditionally. L2CS (trained only on human faces) returns a confident pitch/yaw for *any* crop — animal, poster, or empty background — so non-human and degenerate crops produced high "attention" scores. In the June 3 10-clip smell test (`e2e_reports/2026_06_02_layer03a/`) the single highest-scoring track (`25ffbde8`, score 0.92, target "Camera") was a **dog** the wearer was petting.
   - **Solution**: Added a MediaPipe Tasks API BlazeFace face-presence gate (`models/mediapipe/blaze_face_short_range.tflite`, `min_detection_confidence = MIN_FACE_CONF`, default 0.5) in `src/layer_03a_attention/pipeline.py`. Each candidate crop is checked via `_crop_has_face()` before scoring; crops with no detected human face emit `{"score": 0.0, "target": "NoFace"}` and skip L2CS, so dogs / empty / occluded / unresolvable crops no longer fabricate a gaze score. The gate is env-toggleable (`SR_03A_FACE_GATE`) and threshold-tunable (`SR_03A_MIN_FACE_CONF`); on a missing model/mediapipe or a detector exception it fails open (scores the crop) so a VLM/model outage never silently zeros every bystander. `gaze_target_classification` was updated to exclude `"NoFace"` (as it already excluded `"Unknown"`) so the per-person target reflects gaze on face-present samples. Validated on the June 3 sample: the dog crop is rejected, the post-fix v2 re-run shows 6–95% `NoFace` per clip (reflecting Ego4D's distant/profile bystanders), and the remaining high scorers (`14f5014d`, `6fd026d8`) are confirmed real humans. Residual L2CS gaze noise on hard poses (sunglasses/profile/bent-over) is unchanged but now bounded to genuine human faces (see Model Selection Rationale).

3. **Stale / Drifting Bystander Boxes Under Fast Egocentric Motion (Resolved - June 03)**:
   - **Problem**: Node 02 persists `bystander_detections` at 1/3 FPS (a box every ~3 s, Resolved Issue #11), but 03a samples gaze at 8 FPS and matched each sample to the nearest manifest box within a 2.0 s tolerance. In fast-moving egocentric clips the box was up to 2 s stale, so the gaze crop landed on the wrong region — for `25ffbde8` a single track's box sat on a dog at t=31 s and on empty canyon terrain at t=13 s — yielding meaningless scores and the bulk of the noisy zero/peak trace distribution.
   - **Solution**: Added per-frame YOLO-pose re-detection in `_track_and_score_batched`. On each decoded sample frame, `_detect_pose_boxes()` runs `yolov8n-pose.pt` (resolved via `models_config.get_model("social_presence_pose")`) once and the result is shared across all bystanders active at that timestamp; for each track the stale manifest box is replaced by the maximum-IoU fresh detection when IoU ≥ `REDETECT_IOU_THRESH` (default 0.1), otherwise it falls back to the manifest box. This re-aligns the crop to the subject under motion and synergistically improves the Issue-2 face gate's recall (a fresh, tight box gives the detector a better crop). Env-toggleable via `SR_03A_BBOX_REDETECT`; it adds one detection pass per sampled frame (a material cost on multi-hour clips — disable for throughput-bound batch runs). The model is freed alongside L2CS in `unload()`.

4. **Length-Invariant Engagement Metrics (Resolved - June 03)**:
   - **Problem**: `average_attention_score` (and the `mean_attention_all_persons` aggregate) averages over the entire trace, including the many 0.0 / `NoFace` samples, so it scales with clip length rather than engagement — in the June 3 sample the shortest clip had the highest mean while a 4.7 h clip would regress toward 0 regardless of genuine looking. The summary statistic was a misleading headline for downstream layers (e.g. 03b correlation).
   - **Solution**: Added two additive per-person fields in `process_video`: `attended_fraction` (share of trace samples with score ≥ 0.5 — how often the bystander looked) and `engaged_attention_score` (mean over only score > 0 samples — how intently when they did). The existing `average_attention_score`, `peak_engagement_timestamp_sec`, `sustained_engagement_sec`, and `is_engaged` are retained unchanged (additive schema contract). Post-fix v2 example: `6fd026d8` reports `attended_fraction` 0.23 / `engaged_attention_score` 0.80 against a length-diluted mean of 0.11, making the genuine engagement legible.

5. **Window-Restricted Sampling for Throughput (Resolved - June 03)**:
   - **Problem**: `_track_and_score_batched` sampled the *entire* clip at 8 FPS and ran the per-frame YOLO-pose re-detect (Resolved Issue #3) on every retrieved frame — but only **~19.4%** of clip-time actually contains a bystander detection (measured across the June 3 10-clip sample). The other ~81% of frames were decoded and YOLO'd only to be discarded by the 2.0 s nearest-detection check, making the post-Issue-#3 run **~6.6× slower (9 → 61 min on the 10-clip sample)** and projecting the 1,000-clip reservoir to multiple days.
   - **Solution**: Added window-restricted sampling, gated by `SR_03A_WINDOW_RESTRICT` (default on) with `SR_03A_WINDOW_PAD_SEC` (default 2.0, matching the nearest-detection tolerance). Before the decode loop, `_track_and_score_batched` builds the union of ±`WINDOW_PAD_SEC` intervals around every bystander detection timestamp. Decoding stays **fully sequential and frame-accurate**; for samples that fall in a gap (no bystander within tolerance) the frame is grabbed but the expensive per-frame YOLO re-detect and gaze scoring are skipped (those samples recorded nothing under the 2.0 s check anyway), advancing the active tracks by baseline stride exactly as the in-loop gap path did. An earlier attempt that used `cap.set()` to also skip gap *decode* was reverted: `cap.set` is keyframe-approximate on H.264, desyncing frame↔timestamp and corrupting scores (max mean-score drift 0.38, inflated sample counts). The final no-seek version is **bit-identical to the full-clip output** (max mean-score diff **0.0000**, identical trace-point counts across all 9 validated clips) while running **2.7× faster (61.2 → 22.6 min)** on the 10-clip sample.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: 03a throughput at corpus scale (gap decode + within-window per-frame YOLO)
**Status**: ⚠️ Confirmed Unresolved — Window-restricted sampling (Resolved Issue #5) cut the 10-clip run 2.7× (61 → 23 min), but two costs remain: (a) decode stays fully sequential over the whole clip (gap frames are grabbed, not skipped, to preserve frame-accuracy), and (b) the per-frame YOLO-pose re-detect (Resolved Issue #3) still runs on every sampled frame *within* bystander windows. At the 1,000-clip reservoir (median ~26 min/clip) this still projects to many hours. Three further levers were scoped during the June 3 scaling review and deferred:

**Option A (recommended)**: **Frame-accurate seeking decoder (decord / PyAV / VideoToolbox)** — replace cv2 sequential `grab()` with a decoder supporting exact frame seeking, so gap *decode* (not just gap-YOLO) is skipped.
  - *Pros*: Removes the gap-decode floor that windowing alone cannot; compounds with Resolved Issue #5.
  - *Cons*: New dependency; the cv2 `cap.set` path is keyframe-approximate and was shown to corrupt scores (Resolved Issue #5), so frame-accuracy must be verified on the Ego4D H.264 corpus.

**Option B**: **Cheaper within-window re-detect** — run YOLO-pose at a coarse rate (1–2 FPS) and track between detections (cv2 KCF/CSRT), or re-detect only when the manifest box is stale, instead of every 8 FPS frame.
  - *Pros*: Directly cuts the dominant within-window cost; no new dependency.
  - *Cons*: Tracker drift between detections; must validate that the crop alignment from Resolved Issue #3 is preserved.

**Option C**: **Parallelize / pipeline** — overlap CPU decode with MPS inference and/or process multiple clips concurrently for the decode-bound portion.
  - *Pros*: Multiplies throughput on the 64 GB host.
  - *Cons*: A single MPS GPU serializes inference, so gains are mostly on decode; added concurrency complexity. Best assessed after A/B narrow the bottleneck.

Your selection: _____


