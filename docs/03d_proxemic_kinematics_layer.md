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

### 🐛 Bug Fixes & Refinements

During code audit and validation of the initial implementation, the following issues were identified and resolved:

1. **Depth Map Resolution Mismatch with Bounding Box Coordinates (Resolved - April 28)**:
   - **Problem**: The Depth Anything V2 pipeline returns a depth map that may be a *different resolution* than the input frame (e.g., the model's internal processing size). The original code applied the raw bounding box pixel coordinates from the *original frame* directly onto the depth map array to create the masking region. If the depth map was, for example, 518×518 while the frame was 1920×1080, the mask coordinates would be completely wrong — producing either an out-of-bounds index error or masking the wrong region of the depth map, yielding garbage depth deltas.
   - **Solution**: After obtaining the depth map array, the code now reads its actual `(dh, dw)` dimensions and computes `scale_x = dw / w` and `scale_y = dh / h` to rescale the bounding box coordinates before masking.

2. **`VideoCapture` Resource Leak on `fps == 0` Early Return (Resolved - April 28)**:
   - **Problem**: In `_calculate_depth_delta`, if `cap.get(cv2.CAP_PROP_FPS)` returned 0, the method returned `None` without calling `cap.release()`. On long batch runs, this leaked file descriptors and could eventually trigger `OSError: [Errno 24] Too many open files`.
   - **Solution**: Added `cap.release()` before the early return.

3. **`PIL.Image` Re-imported Inside Hot Loop (Resolved - April 28)**:
   - **Problem**: `from PIL import Image` was placed inside the `for t, bbox in valid_frames` loop in `_calculate_depth_delta`. Python re-resolves the import statement on every iteration, adding unnecessary overhead for every frame processed.
   - **Solution**: Moved the `from PIL import Image` import to the module's top-level imports.

4. **Dead `import math` (Resolved - April 28)**:
   - **Problem**: The `math` module was imported at the top of `pipeline.py` but never used anywhere in the file.
   - **Solution**: Removed the unused import.

### ⚠️ Known Limitations & Troubleshooting

- **Missing `transformers` Dependency**: If you encounter an error stating `transformers and torch are required`, ensure you have installed them in your active virtual environment (`pip install transformers huggingface_hub`).
- **Occlusion Glitches**: Fast-moving objects (like hands or tools) passing in front of the bystander's bounding box can skew the median depth map calculation. The system relies on the YOLO person bounding box to mask the depth map, which does not currently perform instance segmentation (pixel-perfect masking). A future improvement would be to use a SAM-style instance segmentation mask instead of the rectangular YOLO bbox.
- **Adaptive Frame Sampling**: To maintain performance, the depth estimator extracts a maximum of 5 evenly spaced frames from the task reaction window. This prevents GPU/MPS memory exhaustion but may miss micro-movements occurring between sampled frames.
- **Test Suite Fragility (`test_schema_conformance`)**: The `test_schema_conformance` test globally patches `pathlib.Path.exists` to always return `True`. This is necessary to bypass the file-existence check for the dummy video path, but it also affects the resumability guard in `__init__` (which checks if the output file already exists). If tests are run in certain orders or combined with other test files that create real output files, the global patch could cause unexpected behavior. A more surgical mock (patching only the specific `Path` instance) would be safer but is deferred for now as the current test suite runs cleanly.
