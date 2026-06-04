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

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: No human-face validity gate before gaze regression
**Status**: ⚠️ Confirmed Unresolved — Verified in the June 3 10-clip smell test (`e2e_reports/2026_06_02_layer03a/`). The highest-scoring track in the batch (`25ffbde8`, score 0.92, target "Camera") is a **dog** that fills the egocentric frame while the wearer pets it; `_track_and_score_batched` ([pipeline.py](file:///Users/louisye/Desktop/Louis/social_robotics/src/layer_03a_attention/pipeline.py#L306-L503)) crops the Node-02 bystander bbox and runs L2CS-Net on it unconditionally, and L2CS (trained only on human faces) returns a confident pitch/yaw for any crop — dog, mannequin, poster, or empty background. There is no check that the crop actually contains a forward-facing human face, so non-human and degenerate crops produce high "attention" scores. Root cause is shared with Node 02 Issue 2 (animals pass the social gate), but 03a should not blindly trust that every bystander box is a human.

**Option A (recommended)**: **Add a lightweight face-presence + quality gate on each crop** — run a fast face detector (e.g. MediaPipe FaceDetection / BlazeFace, already an MPS-friendly dependency) on the crop before L2CS; if no human face clears a confidence + min-pixel-size threshold, emit `score=0.0, target="NoFace"` for that sample instead of a gaze score.
  - *Pros*: Directly removes the dog/empty/poster false-highs; cheap (BlazeFace is sub-millisecond); also fixes the distant/occluded-face over-confidence (faces below the min-size threshold are gated out); adds an auditable `NoFace` reason to the trace.
  - *Cons*: Adds one detector forward pass per crop (mitigated by the existing batching); a too-strict size threshold could drop genuine distant bystanders (needs tuning against the pass-score distribution).

**Option B**: **Confidence-weight the score by L2CS face-embedding plausibility** — keep scoring all crops but down-weight samples whose L2CS internal feature confidence is low.
  - *Pros*: No second model; single-pass.
  - *Cons*: L2CS does not expose a calibrated face-confidence; brittle and indirect; would not cleanly reject a dog face that L2CS is "confident" about.

**Option C**: **Fix it upstream only (rely on Node 02 Issue 2 remediation)** — assume a human-validated bystander gate at Node 02 and make no 03a change.
  - *Pros*: Single chokepoint; no per-frame cost in 03a.
  - *Cons*: Couples 03a correctness to a 02 change that is itself unresolved; 03a remains wrong for any manifest produced before that lands, and for any future non-human that slips the 02 gate.

Your selection: Proceed with Option A.

---

### Issue 2: Gaze crops use stale, drifting Node-02 bystander boxes under fast egocentric motion
**Status**: ⚠️ Confirmed Unresolved — Verified in the June 3 smell test. Node 02 persists `bystander_detections` at 1/3 FPS (one box every ~3 s, per Resolved Issue #11), but 03a samples gaze at 8 FPS and matches each sample to the nearest box within a **2.0 s** tolerance ([pipeline.py](file:///Users/louisye/Desktop/Louis/social_robotics/src/layer_03a_attention/pipeline.py#L365-L392)). In fast-moving egocentric clips the box is up to 2 s stale, so the crop lands on the wrong region: for clip `25ffbde8` a single `person_id` track's box sits on a dog at t=31 s and on **empty canyon terrain** at t=13 s (the real hikers are center-frame, outside the box), indicating both 2 s staleness and ByteTrack ID drift across the moving camera. Stale crops yield meaningless gaze scores (high when the crop accidentally catches a face, zero when it catches background), which is the dominant source of the 43%-zero / noisy-peak trace distribution.

**Option A (recommended)**: **Re-detect the bystander box at the gaze sample time** — run the cheap YOLO-pose detector (already used by Node 02) on the sampled frame and associate the nearest box to the track before cropping, rather than reusing a box up to 2 s old.
  - *Pros*: Eliminates staleness; crops follow the subject under fast motion; reuses an existing model.
  - *Cons*: Adds a detection pass per sampled frame (cost partly offset by skipping samples with no nearby box); needs a track-association rule to keep `person_id` stable.

**Option B**: **Tighten the match tolerance and interpolate** — drop the tolerance from 2.0 s to ~0.4 s and linearly interpolate box position between the two nearest Node-02 detections.
  - *Pros*: No new model; interpolation smooths slow motion.
  - *Cons*: Linear interpolation is wrong under fast/jerky egocentric motion (the failure case here); a tight tolerance without interpolation would simply drop most 8-FPS samples, collapsing the trace toward the 1/3-FPS box cadence.

**Option C**: **Raise Node 02's bystander sampling rate** for retained clips so boxes exist near every 03a sample.
  - *Pros*: Fixes staleness at the source for all downstream layers.
  - *Cons*: Re-running Node 02 over the 1,000-clip reservoir is expensive; increases manifest size; the 1/3-FPS rate was deliberately chosen for 02 throughput (Resolved Issue #11).

Your selection: Proceed with Option A.

---

### Issue 3: `average_attention_score` compresses toward a low band and tracks clip length, not engagement
**Status**: ⚠️ Confirmed Unresolved — Verified in the June 3 smell test. All 10 clips scored a mean of 0.16–0.41 (none ≥ 0.5), yet every clip's **peak** single-sample score was 0.80–1.00 and 21% of all 50,502 samples were ≥ 0.70. Because `mean_attention_all_persons` averages over the full trace — 43% of which are exactly 0.0 (subject not looking, or stale/empty crop per Issue 2) — the summary statistic is dominated by clip length: the shortest clip (`25ffbde8`, 122 s) has the highest mean and a 4.7 h clip would regress toward 0 regardless of genuine engagement. The per-person `is_engaged` flag partly compensates via the `sustained_engagement_sec > 2.0` clause, but `average_attention_score` as exposed is a misleading headline metric for downstream layers (e.g. 03b correlation).

**Option A (recommended)**: **Report engagement over "attended" samples, not all samples** — add `attended_fraction` (share of samples with score ≥ 0.5) and `engaged_attention_score` (mean over only score>0 samples) alongside the existing mean; document the peak/sustained metrics as the primary engagement signal.
  - *Pros*: Length-invariant; separates "how often they looked" from "how intently when they did"; additive schema change, no recompute of traces.
  - *Cons*: More fields for downstream layers to understand; thresholds (0.5) need a short calibration pass.
  - *Note*: Most informative once Issue 2 is fixed, since today's zero-samples are a mix of genuine not-looking and stale-crop artifacts.

**Option B**: **Keep the mean but window it** to the task reaction-window (from `task_temporal_metadata`) instead of the whole clip.
  - *Pros*: Focuses the score on the moment that matters for reaction-class analysis; aligns 03a with the climax-window design.
  - *Cons*: Requires climax metadata to be populated first (deferred to the first Layer 03 pass per 02 Resolved Issue #8); undefined for clips with no clear climax.

Your selection: Proceed with Option A.


