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

11. **Dead `tempfile.TemporaryDirectory` Wrapper Around Track-and-Score Loop (Resolved - May 07)**:
    - **Problem**: `_track_and_score` opened a `with tempfile.TemporaryDirectory() as temp_dir:` context that wrapped the entire per-bystander scoring loop, bound the path to `temp_path`, and then never used it. This was a leftover from an earlier implementation that wrote cropped face frames to disk for VLM spot-checks. Every bystander invocation paid the syscall cost of creating and tearing down a system temp directory for zero benefit.
    - **Solution**: Removed the `import tempfile`, deleted the `with tempfile.TemporaryDirectory()` wrapper, and un-indented the loop body. Behavioral output is unchanged.

12. **Adaptive 8/16 FPS Sampling Restoration (Resolved - May 07, supersedes prior fixed-8 FPS Resolved Issue + the previous "dead `last_score`" finding)**:
    - **Problem**: The implementation diverged from the documented adaptive sampling strategy in two ways. (a) The doc described a tiered 2 FPS / 5 FPS adaptive scheme; the code ran at a fixed 8 FPS stride and emitted `"fixed_8fps"` as a string in `processing_meta.sampling_fps_effective`, breaking numeric parsing for downstream consumers. (b) `last_score` was initialized and assigned each loop but never *read*, after adaptive sampling had been removed in a prior pass. A separate audit had flagged the variable for deletion ("dead variable" finding), but the user-selected resolution path was instead to *resurrect* adaptive sampling rather than delete the variable, since downstream Layer 03e's Nyquist analysis only requires an 8 FPS *floor*, not a uniform stride.
    - **Solution**: Reintroduced adaptive sampling with parameters chosen to preserve the 03e Nyquist contract. Baseline stride is `0.125 s` (8 FPS, Nyquist = 4 Hz), and on `|score - last_score| > 0.3` the loop boosts to `0.0625 s` (16 FPS) for the next 2 s before decaying back. `last_score` is now actively read each iteration to drive the burst trigger, and the burst-window timer (`burst_until_t`) makes the decay deterministic. `processing_meta` now emits numeric `sampling_fps_effective: 8.0`, `sampling_fps_burst: 16.0`, and a descriptive `sampling_strategy: "adaptive_8_to_16_fps"`. Section 2 of this document and the output schema example were rewritten to match. A note was added to the schema clarifying that `attention_trace[i].t` is authoritative and downstream filters that require uniform sampling (e.g., 03e's `filtfilt`) must resample onto a fixed-dt grid first — which 03e already does.

13. **Per-Sample Gaze Target Attribution (Resolved - May 07)**:
    - **Problem**: `gaze_target_classification` in the per-person output was hardcoded to the placeholder string `"Camera_or_Task"` regardless of which raycast geometry actually maximized the dot product on a given frame. The schema example in the docs advertised dynamic values like `"POV_Actor_Hands"`, so the contract was silently broken for every consumer that tried to differentiate face-directed vs. task-directed attention.
    - **Solution**: `_track_and_score` now records a per-sample `target` label of `"Camera"`, `"POV_Actor_Hands"`, or `"Unknown"` based on which dot-product won (`max_dot_prod` vs. each candidate). The label is appended to every `attention_trace[i]` entry. `process_video` then aggregates the majority non-`"Unknown"` `target` across the trace and uses that as `gaze_target_classification` for the per-person summary. The schema and contract paragraph in the doc were updated accordingly so consumers can opt into per-sample granularity via `attention_trace[i].target` or stay with the per-person majority.

14. **Unsafe `cap.retrieve()` on First Iteration (Resolved - May 07)**:
    - **Problem**: `_track_and_score` set the capture cursor to frame 0 with `cap.set(CAP_PROP_POS_FRAMES, 0)` and entered its scoring loop. On the first iteration `target_frame_idx == 0 == current_frame_idx`, so the inner `while current_frame_idx < target_frame_idx: cap.grab()` loop never ran. Reaching `cap.retrieve()` *without* a prior `cap.grab()` is undefined behavior in OpenCV — depending on the backend and codec, the buffer could hold a stale frame, an empty/black frame, or fail silently. On macOS / AVFoundation this manifested as an unpredictable first-frame pitch/yaw read.
    - **Solution**: Added an explicit priming step immediately after `cap.set(...)`: a single `cap.grab()` is issued and `current_frame_idx` is initialized to `1`. The loop's skip-and-retrieve logic continues to work for all subsequent samples, and the inner loop's invariant (`current_frame_idx >= target_frame_idx` before retrieve) is now enforced from the very first iteration. If the prime fails (e.g., 0-frame video) the function returns an empty trace cleanly instead of producing garbage data.

15. **Bare `except:` Clauses Hid Interrupts and OOM (Resolved - May 07)**:
    - **Problem**: `AttentionLayerPipeline.run` and `log_error` each contained a bare `except:` that swallowed *every* exception when loading existing JSON state or the error log. On long batch runs on the M4 Pro this meant `Ctrl+C` was eaten during the file-load phase and `MemoryError` from MPS pressure was silently masked, leaving the operator with no signal about what went wrong.
    - **Solution**: Replaced both bare clauses with `except (json.JSONDecodeError, IOError, ValueError):`. `KeyboardInterrupt`, `SystemExit`, and `MemoryError` now propagate to the caller, while genuinely-expected I/O and parsing failures still let the pipeline fall back to a fresh state.

16. **L2CS Model / MPS Memory Unloading via Context Manager (Resolved - May 07)**:
    - **Problem**: `AttentionLayerPipeline` loaded the L2CS-Net ResNet50 graph into MPS memory at construction and held onto it for the lifetime of the Python process. There was no `unload()` method, no `__del__`, and no context-manager support, so when the E2E pipeline transitioned from Layer 03a to Layers 03b–03g the ~200 MB of L2CS weights remained resident and competed with the next layer's models for unified memory on the 24 GB Mac mini M4 Pro.
    - **Solution**: Added an explicit `unload()` method that deletes `self.gaze_pipeline`, runs `gc.collect()`, and clears the MPS / CUDA cache (mirroring the `SocialPresenceDetector.unload()` pattern). Implemented `__enter__` / `__exit__` on `AttentionLayerPipeline` so callers can write `with AttentionLayerPipeline(...) as pipeline: pipeline.run()` and get deterministic cleanup on normal exit *or* on exceptions. Direct instantiation continues to work for short-lived scripts and for the existing test suite — callers that don't need cleanup can ignore the protocol.

17. **ByteTrack State Contamination Across Videos in 03a's Upstream (Resolved - May 07)**:
    - **Problem**: Layer 03a consumes pre-computed `bystander_detections` from Node 02's `SocialPresenceDetector`, which calls `model.track(..., persist=True)` and reused a single detector instance across every video in the batch. ByteTrack therefore retained Kalman state, track IDs, and IoU association history across video boundaries, which could (a) cause `person_id` values to monotonically increment across the dataset, (b) generate false re-associations when a bystander in video N+1 appeared near where a bystander in video N was last seen, and (c) keep ghost tracks alive longer than the intended single-clip horizon. All of those defects propagated downstream into 03a's per-person attention traces.
    - **Solution**: Resolved upstream in `src/shared/social_presence.py:detect()` (see Node 01 Resolved Issue #14, Node 02 Resolved Issue #15). The detector now walks `self.model.predictor.trackers` and calls `.reset()` on each tracker at the top of every `detect()` invocation, guarded by a try/except for Ultralytics API drift. Layer 03a inherits the fix without any per-layer change, since it consumes the manifest emitted by Node 02 rather than running tracking itself.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: L2CS-Net `unload()` Pattern Now Premature on 64 GB Mac Studio
**Status**: ⚠️ Confirmed Unresolved — Resolved Issue #16 ("L2CS Model / MPS Memory Unloading via Context Manager") added an explicit `unload()` method and a `__enter__/__exit__` protocol so the E2E orchestrator could free L2CS-Net's ~200 MB ResNet50 graph (plus ~500 MB of cached MPS activations) before transitioning to Layers 03b–03g. The driver was the 24 GB Mac mini M4 Pro's inability to keep L2CS resident alongside Py-Feat (03b), emotion2vec+ (03c), and Depth Anything V2 + SAM (03d). On the new **Mac Studio (M4 Max, 64 GB unified memory)** host, the steady-state combined resident set of L2CS + every downstream 03 model totals ~12 GB — well under the 40 GB budget. Forcing an unload now means the E2E orchestrator pays a ~1.5 s reload + MPS warm-up if any downstream layer needs to re-query 03a's gaze model (e.g., a re-run, a `--force` reprocess), with no compensating memory win.

**Option A (recommended)**: **Keep `unload()` and `__exit__` But Make the E2E Orchestrator Opt-In** — The context manager and `unload()` method stay in `AttentionLayerPipeline` (they're useful for short-lived scripts and for memory-constrained hosts). The E2E orchestrator stops calling `unload()` automatically when `psutil.virtual_memory().total >= 48 * 2**30` and instead retains the pipeline instance for the entire 03 run. Direct callers using `with AttentionLayerPipeline(...) as p:` still get deterministic cleanup.
  - *Pros*: Preserves Resolved Issue #16's contract for short-lived/constrained callers; eliminates the reload-cost regression on the M4 Max; the gate is a one-line check.
  - *Cons*: Adds a host-class branch in the E2E orchestrator; if the orchestrator forgets to call `unload()` on exit, a long-lived L2CS instance survives until process exit (acceptable on the M4 Max but should be documented).

**Option B**: **Remove `unload()` Entirely; Always Hold L2CS Resident** — Treat the M4 Max as the supported host and delete the `unload()` method and context-manager protocol.
  - *Pros*: Smallest 03a codebase; one resident-set policy.
  - *Cons*: Hard-breaks Mac mini compatibility; loses the operationally-validated cleanup path Resolved Issue #16 added; closes the door on memory-stressed CI runners.

Your selection: _____

---

### Issue 2: L2CS-Net (ResNet50 Backbone) Predates Stronger MPS-Compatible Gaze Models
**Status**: ⚠️ Confirmed Unresolved — `L2CS-Net` was selected (Resolved Issue #1) because its ~200 MB ResNet50 backbone fit comfortably in the 24 GB Mac mini's MPS-resident budget and because the upstream `models/l2cs-net/` repo could be patched in-tree for the `include_detector=False` path. The model is from 2022 and its angular error on Gaze360 (~10°) is now eclipsed by **3DGazeNet** and **CrossGaze** (~7° on Gaze360), both Apache-2.0 and both with ~500 MB-1 GB checkpoints that would not have fit alongside the rest of the 03 layers on a 24 GB host but easily fit on the **Mac Studio (M4 Max, 64 GB unified memory)** target.

**Option A (recommended)**: **A/B CrossGaze Against L2CS-Net on a 100-Clip Validation Subset** — Add `models/crossgaze/` as a sibling to `models/l2cs-net/`, wire it through the same `gaze_pipeline` interface used by `AttentionLayerPipeline`, and run both models on the same 100-clip subset. Compare per-trace `pitch_rad`/`yaw_rad` against ground-truth Aria-glasses gaze in the subset of Ego4D clips that carry it. Promote CrossGaze as the default only if the angular error drop exceeds 1.5° and Layer 03e's nod/shake F1 score on the corresponding subset does not regress.
  - *Pros*: Direct measurable quality target (angular error); the 03e regression check guards the most sensitive downstream consumer of L2CS's pitch/yaw output; no production change until measured.
  - *Cons*: Requires standing up a second `gaze_pipeline` adapter and a comparison harness; CrossGaze's coordinate convention may not match `gazeto3d` (Resolved Issue #2) so the per-frame integration path needs validation; ~1-2 weeks of effort before merge.

**Option B**: **Hold on L2CS-Net Until a Specific 03 Downstream Bug Demands More Precision** — Defer the upgrade until a concrete failure mode (e.g., 03e false-shake-detection that traces back to noisy L2CS pitch/yaw) justifies the effort.
  - *Pros*: Zero risk of introducing a coordinate-convention or calibration regression in the gaze pipeline; engineering time stays on higher-priority issues.
  - *Cons*: Forfeits the headroom advantage of the new host; if a future audit asks "why are you on a 4-year-old model," there's no documented reason beyond inertia.

**Option C**: **Switch to a VLM-Based Gaze Estimator (Qwen2.5-VL or Gemma 4 Multimodal)** — Replace L2CS with a prompt-based "estimate the bystander's gaze pitch/yaw" call against the existing VLM stack.
  - *Pros*: Reuses already-resident VLM weights; no second backbone to maintain.
  - *Cons*: VLM gaze estimation is currently 5-15° worse than dedicated regression models; latency is ~20× higher per frame; doesn't produce continuous radians, only discrete classifications — would break Layer 03e's frequency-domain consumption entirely.

Your selection: _____

