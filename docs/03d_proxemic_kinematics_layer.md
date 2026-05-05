# AI Task Breakdown: Proxemic Kinematics Layer (03d)

## Objective
The **Proxemic Kinematics Layer** investigates "Approach vs. Avoidance." In developmental stages, an infant assesses the severity of an action by how the caregiver physically manages space (Proxemics). If a child touches something dangerous, the caregiver lunges forward (Approach/Intervention). If they smell something rotten, the caregiver recoils (Avoidance). This layer measures the physical vector of the bystander relative to the POV camera.

---

## 📥 Input Requirements
- **`filtered_manifest.json`**: Needs the `bystander_detections` array containing the bounding box coordinates of the other person across time, and the `task_reaction_window_sec`.
- **Cross-layer (optional)**: None strictly required, though it correlates heavily with motor resonance.

---

## 🛠️ Implementation Strategy

We can track depth/proximity changes via two complementary methods:

### 1. Bounding Box Scaling (Fast / Heuristic)
Track the area $A = (x_{max} - x_{min}) \times (y_{max} - y_{min})$ of the bystander's bounding box precisely during the reaction window.
- **Rapid Expansion**: The bounding box grows exponentially relative to the frame size. This indicates a forward lunge or approach.
- **Rapid Contraction**: The bounding box shrinks. The bystander is stepping back or recoiling away from the POV actor.

### 2. SOTA Monocular Metric Depth (Depth Anything V2-Small)
For an extremely accurate vector, we compute the relative Z-distance using a SOTA foundational depth model.
- **Model**: **Depth Anything V2-Small** (`vits`, ~25M params, **Apache-2.0**). Fallback: **Depth Anything V1-Large** (`vitl`, ~335M params, also Apache-2.0) if V2-Small quality is insufficient.
- **Why V2-Small**: The pipeline tracks **relative depth deltas** (∆Z between frames), not absolute metric depth. V2-Small's relative ordering is sufficient for detecting approach vs. retreat. Larger V2 variants (Base/Large/Giant) are CC-BY-NC-4.0 and would restrict the exported dataset.
- **Mechanism**: Run the depth model at 2-3 FPS over the reaction window. Mask the depth map using the YOLO `person` bounding box to isolate the bystander's pixels.
- **Calculation**: Calculate the median depth value of the bystander mask. Track the $\Delta$ depth over time. A rapidly decreasing depth map value indicates approach.

### 3. Proxemic Vector Formulation
Combine the scale and depth delta into a normalized `proxemic_vector` ranging from -1.0 (hard avoidance/recoil) to +1.0 (hard approach/intervention).

---

## 📤 Output Schema and Integration
**Example Output Data (`03d_proxemic_kinematics_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03d_proxemic_kinematics",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "per_person": [
        {
          "person_id": 0,
          "bbox_scale_delta_pct": 24.5,
          "depth_anything_v2_delta": -0.32,
          "proxemic_vector": 0.85,
          "classified_action": "Approach_Intervention"
        }
      ]
    }
  ]
}
```

## Verification & Validation Check
- **Singular Video Test**: Process a video where a bystander walks towards the camera. Overlay the Depth Anything V2-Small depth-map as a colormap mask next to the YOLO bounding box. Verify visually that the assigned Z-median steadily decreases as the person approaches.
- **Batch Test**: Run across a batch of standard interaction videos. Validate that `Depth Anything V2-Small` tensor offloading functions correctly on the **Mac mini M4 Pro (24GB RAM)** via the PyTorch MPS backend. Assert that the `proxemic_vector` appropriately penalizes jitter (ignoring +/- 0.05 micro-movements to avoid false positive "lunges"). If V2-Small accuracy proves insufficient, fall back to Depth Anything V1-Large (still Apache-2.0).

## 🚀 Implementation Accomplishments

The 03d Proxemic Kinematics layer has been fully implemented with the following features:

- **Dual-Heuristic Vector**: The `proxemic_vector` is computed using a weighted combination of bounding box scale delta (40% weight) and Depth Anything V2 median depth delta (60% weight).
- **Extreme SSD Caching**: To prevent filling the internal drive, the Hugging Face cache for `transformers` is programmatically locked to `/Volumes/Extreme SSD/huggingface_cache` during pipeline initialization. The model weights are automatically downloaded to this location on first run.
- **Strict Model Enforcement**: The bounding box fallback has been removed per architecture review. The pipeline now explicitly requires `transformers` and `torch` to be installed and will raise a `RuntimeError` if the Depth Anything V2 model cannot be initialized.
- **Resilient Batch Processing**: Errors during depth map generation or video extraction are safely caught, logged to `03d_proxemic_kinematics_errors.json`, and the pipeline gracefully continues to the next video.

## 🧪 Resolved Issues & Implementation Refinements

1. **Depth Map Resolution Mismatch (Resolved - April 28)**:
   - **Problem**: The Depth Anything V2 pipeline returns depth maps at model-specific resolutions (e.g., 518x518), causing frame-level bounding box coordinates (e.g., 1920x1080) to index out-of-bounds or mask incorrect regions.
   - **Solution**: Implemented dynamic rescaling of bounding box coordinates based on the ratio between the original frame dimensions and the returned depth map dimensions.

2. **`VideoCapture` Resource Leak (Resolved - April 28)**:
   - **Problem**: An early return in `_calculate_depth_delta` (triggered when `fps == 0`) failed to call `cap.release()`, leading to file descriptor leakage and potential `OSError: [Errno 24]` during large batch runs.
   - **Solution**: Added explicit `cap.release()` calls before all early return paths.

3. **Performance Degradation via Inline Imports (Resolved - April 28)**:
   - **Problem**: `from PIL import Image` was re-imported inside the frame processing hot loop, adding redundant overhead for every sampled frame.
   - **Solution**: Moved all `PIL` imports to the module's top-level scope.

4. **Dead Code Removal (Resolved - April 28)**:
   - **Problem**: Unused `import math` remained in `pipeline.py`, increasing module load time and cluttering the codebase.
   - **Solution**: Removed the unused import.

5. **HuggingFace Cache Location Override (Resolved - April 29)**:
   - **Problem**: Setting `os.environ['HF_HOME']` inside the class constructor was too late, as the `transformers` library initializes its cache paths upon module-level import.
   - **Solution**: Moved the `HF_HOME` environment assignment to the absolute top of `pipeline.py`, ensuring it precedes any `transformers` or `torch` imports.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Occlusion Glitches (Bounding Box vs. Instance Mask)
**Status**: ⚠️ Confirmed Unresolved — The pipeline masks the depth map using the rectangular YOLO `person` bounding box. When objects pass between the camera and the bystander (e.g., hands, tools, furniture), the bounding box captures these occluding pixels, skewing the median depth value towards the camera. This causes false "approach" spikes in the `proxemic_vector`.

**Option A (recommended)**: **SAM 2 Instance Segmentation Mask** — Run [SAM 2](https://github.com/facebookresearch/segment-anything-2) (Segment Anything Model 2) on the bystander's bounding box to generate a pixel-precise person mask. Use this mask instead of the rectangular bounding box to filter the depth map. SAM 2 is Apache-2.0 licensed and supports video propagation (mask-once, track-through-clip).
  - *Pros*: Eliminates occluding object pixels from depth computation; SAM 2's video propagation avoids per-frame segmentation cost; significantly more accurate `proxemic_vector`.
  - *Cons*: SAM 2 adds ~150-400MB model weight depending on variant; first-frame segmentation takes ~200ms; increases pipeline complexity; must handle cases where SAM fails to segment the person.

**Option B**: **Depth-Based Outlier Rejection** — Instead of masking by pixel identity, reject depth values within the bounding box that deviate >2σ from the median. This statistically removes occluding objects (which are at different depths) without needing instance segmentation.
  - *Pros*: Zero additional model dependencies; trivial to implement (~5 lines of numpy); no memory cost.
  - *Cons*: Fails when occluding objects are at similar depth to the bystander; σ threshold requires tuning; less precise than learned segmentation.

Your selection: Proceed with Option A.

---

### Issue 2: Micro-movement Aliasing (Adaptive Sampling)
**Status**: ⚠️ Confirmed Unresolved — The pipeline samples a maximum of 5 frames per reaction window using `cap.set()` random seeking. For reaction windows of 0.5-2.0s (fast tasks), this yields a frame every 100-400ms, which is sufficient. For 2.0-6.0s windows (slow tasks), the 5-frame limit results in sparse coverage (>1s gaps), potentially missing rapid approach/retreat events within the window.

**Option A (recommended)**: **Dynamic Frame Count Based on Window Length** — Scale the number of sampled frames proportionally to the window duration: `num_frames = max(5, int(window_duration * 3))`, capping at 20 frames. This ensures at least 3 FPS coverage regardless of window length while avoiding excessive frame counts.
  - *Pros*: Automatically adapts to window duration; no model changes; preserves existing seeking logic.
  - *Cons*: Longer windows produce more depth inference calls (up to 4x current maximum); may increase per-video processing time from ~2s to ~8s.

**Option B**: **Sequential Frame-Skip Loop** — Replace random seeking with the same sequential `grab()/retrieve()` pattern recommended for 03a Issue 4. Process every Nth frame within the window, where N is computed from the target FPS (3 FPS).
  - *Pros*: Eliminates seeking overhead; consistent temporal spacing; can be combined with Option A's dynamic count.
  - *Cons*: Requires reading all frames between start and end of window (even if skipping most); slightly more complex implementation.

Your selection: Proceed with Option A.

---

### Issue 3: Test Suite Resumability Guard
**Status**: ⚠️ Confirmed Unresolved — The `test_schema_conformance` test globally patches `pathlib.Path.exists` to always return `True`, which interferes with the pipeline's resumability logic (which checks `self.output_result_path.exists()` to skip already-processed videos). If tests are run concurrently or in sequence without cleanup, the global patch can leak into subsequent test methods.

**Option A (recommended)**: **Instance-Level Mock** — Replace the global `@patch("pathlib.Path.exists")` with a targeted `@patch.object` on the specific `Path` instance used by the pipeline's output path. Alternatively, use `tmp_path` fixtures to create actual temporary files that satisfy the `exists()` check without mocking.
  - *Pros*: Eliminates side-effect leakage; tests are isolated and reproducible; follows pytest best practices.
  - *Cons*: Requires refactoring the test setup to inject paths; slightly more verbose test code.

Your selection: Proceed with Option A.

---

### Issue 4: Heuristic Signal Conflict
**Status**: ⚠️ Confirmed Unresolved — The `proxemic_vector` is a weighted combination of bounding box scale delta (40%) and depth delta (60%). In scenarios where the camera and bystander move simultaneously (e.g., camera pans right while bystander walks left), the scale delta may indicate "approach" (bbox growing due to camera pan) while the depth delta indicates "retreat" (bystander moving away). The weighted average produces a near-zero `proxemic_vector` that masks both signals.

**Option A (recommended)**: **Signal Agreement Confidence Score** — Add a `proxemic_confidence` field to the output schema. Compute it as `1.0 - abs(sign(scale_delta) - sign(depth_delta)) / 2`. When both signals agree (same sign), confidence = 1.0. When they disagree (opposite signs), confidence = 0.0. Downstream consumers can filter on this field to exclude ambiguous results.
  - *Pros*: Non-breaking schema addition; gives downstream layers a quality signal; trivial to compute.
  - *Cons*: Does not resolve the ambiguity itself — just flags it; consumers must implement their own handling for low-confidence results.

**Option B**: **EgoMotion Compensation** — Compute the camera's own egomotion (using the same optical flow technique as 03f Motor Resonance) and subtract it from the bounding box scale delta before combining with depth. This isolates the bystander's true proxemic movement from camera-induced scale changes.
  - *Pros*: Resolves the root cause of signal conflict; produces a more accurate `proxemic_vector`; reuses existing optical flow infrastructure.
  - *Cons*: Adds computational overhead (~30% more per video); requires careful calibration of the egomotion subtraction; may introduce its own errors if optical flow is noisy.

Your selection: Proceed with Option B. Make sure to add validation tests or add a confidence score for optical flow noise. If the noise is too much, do not process that video for proxemic_vector. 

---

### Issue 5: Missing Dependency Validation (User-Friendly Startup Check)
**Status**: ⚠️ Confirmed Unresolved — The pipeline raises a generic `RuntimeError` if `transformers` or `torch` are missing, but the error message does not provide actionable remediation instructions. Users encountering this error must manually research which packages to install and which versions are compatible.

**Option A (recommended)**: **Actionable Error Messages with Install Commands** — Replace the generic `RuntimeError` with a detailed message including the exact `pip install` command and minimum version requirements. Example: `RuntimeError: Missing required dependency 'transformers>=4.35.0'. Install with: pip install transformers>=4.35.0 huggingface_hub`.
  - *Pros*: Zero-friction for new users; self-documenting; copy-pasteable fix.
  - *Cons*: Install commands may become stale if version requirements change; does not handle conda environments.

Your selection: Proceed with Option A.
