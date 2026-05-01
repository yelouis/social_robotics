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

- **Occlusion Glitches**: Fast-moving objects passing between the camera and bystander can skew the median depth calculation. Transitioning from rectangular YOLO bounding box masks to pixel-perfect **Segment Anything (SAM)** instance masks is suggested for higher precision.
- **Micro-movement Aliasing**: The current adaptive sampling (max 5 frames per window) may miss rapid micro-movements. Implementing a sequential frame-skipping loop instead of random seeking is recommended to improve temporal resolution without exceeding memory limits.
- **Test Suite Resumability Guard**: The `test_schema_conformance` mock globally patches `pathlib.Path.exists`, which can interfere with the pipeline's resumability logic during concurrent test runs. Moving to a more surgical, instance-level mock is suggested.
- **Heuristic Signal Conflict**: In scenarios with simultaneous camera and bystander movement, scale and depth deltas can contradict each other. Implementing a "Confidence" score based on signal agreement is suggested to flag ambiguous results.
- **Missing Dependency Validation**: The pipeline requires `transformers` and `torch`. Adding a more robust startup dependency check that offers one-line install commands (e.g., `pip install transformers huggingface_hub`) is recommended.
