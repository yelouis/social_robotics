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
- **Adaptive Sampling**: To prevent micro-movement aliasing and temporal gaps, frames are dynamically sampled proportionally to the window duration at ~3 FPS (capped at 20 frames).

### 2. SOTA Monocular Metric Depth
For an extremely accurate vector, we compute the relative Z-distance using a SOTA foundational depth model.
- **Tier-Per-Host Registry**: The pipeline scales automatically based on host memory. It uses **Depth Anything V1-Large** + **SAM ViT-Huge** on 64GB Mac Studio hosts, falling back to **Depth Anything V2-Small** + **SAM ViT-Base** on 24GB hosts.
- **Bbox-prompted SAM Instance Masking**: We use single-pass bbox-prompting via `SamModel` to precisely segment the bystander silhouette, selecting the best mask via the SAM IoU score head. This drastically reduces occlusion noise from foreground objects compared to naive YOLO bounding box crops or expensive automatic mask generation.
- **Calculation**: Calculate the median depth value of the bystander mask. Track the $\Delta$ depth over time using a linear-regression "slope-span" calculation rather than endpoint deltas. This ensures intermediate frame samples contribute to the vector and mitigates endpoint outlier noise.

### 3. Proxemic Vector Formulation
Combine the scale and depth delta into a normalized `proxemic_vector` ranging from -1.0 (hard avoidance/recoil) to +1.0 (hard approach/intervention).
- **Optical Flow Noise Rejection**: Farneback optical flow extracts ego-motion noise. If the 95th percentile magnitude exceeds the noise threshold (indicating extreme camera panning), the proxemic vector is zeroed to prevent false positives.
- **Proxemic Confidence**: Calculated dynamically based on the sign alignment between the bounding box heuristic and the depth delta heuristic.

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
          "classified_action": "Approach_Intervention",
          "proxemic_confidence": 1.0,
          "optical_flow_noise": 5.2
        }
      ]
    }
  ]
}
```

## Verification & Validation Check
- **Singular Video Test**: Process a video where a bystander walks towards the camera. Overlay the depth-map as a colormap mask next to the YOLO bounding box. Verify visually that the assigned Z-median steadily decreases as the person approaches.
- **Batch Test**: Run across a batch of standard interaction videos. Validate that tensor offloading functions correctly on the **Mac Studio (M4 Max, 64 GB unified memory)** via the PyTorch MPS backend. Assert that the `proxemic_vector` appropriately penalizes jitter (ignoring +/- 0.05 micro-movements to avoid false positive "lunges").

## 🚀 Implementation Accomplishments

The 03d Proxemic Kinematics layer has been fully implemented with the following features:

- **Dual-Heuristic Vector**: The `proxemic_vector` is computed using a weighted combination of bounding box scale delta (40% weight) and Depth Anything median depth delta (60% weight).
- **Centralized Tuning Surface**: Proxemic heuristics (noise thresholds, normalization factors, weights, deadbands) are exposed as class-level constants, allowing surgical ablations via single-line subclass overrides.
- **Tier-Per-Host Model Registry**: Dynamic identifiers query `models_config.py` to select depth and SAM variants appropriate for the host tier (e.g., 64GB vs 24GB unified memory), maximizing pipeline fidelity without risking OOM errors.
- **Extreme SSD Caching**: To prevent filling the internal drive, the Hugging Face cache for `transformers` is programmatically locked to `/Volumes/Extreme SSD/huggingface_cache` during pipeline initialization. A strict `os.path.ismount` guard ensures the pipeline fails fast if the external SSD is disconnected.
- **Tiered Accelerator Cache Management**: The pipeline automatically flushes accelerator cache (`torch.mps.empty_cache()`) based on host memory limits (every video for <48GB hosts, every 25 videos for >=48GB hosts), bounding pressure during long batches.
- **Resilient Batch Processing**: Errors during depth map generation or video extraction are safely caught, logged to `03d_proxemic_kinematics_errors.json`, and the pipeline gracefully continues.
- **Resumability & Sentinel Records**: Videos that legitimately produce no output (e.g., filtered tasks, no bystanders) emit sentinel records to the output JSON, preventing the resumability logic from redundantly re-executing expensive depth and optical flow scans.
