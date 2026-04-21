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
