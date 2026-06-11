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

## 🧪 Resolved Issues & Implementation Refinements

1. **Reaction-Window vs Detection-Cadence Geometry Guaranteed 0 Yield (Resolved - June 10)**:
   - **Problem**: The June 9 50-clip smell test (`e2e_reports/2026_06_09_layer03d_50/`) scored **0/50 — every record a `no_output_produced` sentinel**. `_calculate_bbox_scale_delta` / `_calculate_depth_delta` require **≥ 2 bystander detections strictly inside `task_reaction_window_sec`**, but Node-02 detections arrive at a **median 6.0 s cadence** while reaction windows are **2.0 s wide** — a window geometrically holds at most one detection, so ≥ 2 was impossible by construction (0/59 tasks qualified). Compounding it, the window is anchored to the *wearer's* optical-flow climax while the nearest bystander detection sat a median 6.8 s (max 158 s) away — the same wearer-vs-bystander mismatch 03b fixed in its Resolved #2.
   - **Solution** (Option A): Added `_bystander_measurement_window()` (`src/layer_03d_proxemic_kinematics/pipeline.py`): the strict reaction window is kept when it already holds ≥ 2 detections (dense-detection behavior unchanged); otherwise the measurement interval re-anchors to the bystander's climax-nearest detection ± `ANCHOR_SPAN_DETECTIONS` (= 1), a 2–3-consecutive-detection span (~6–12 s nominal). Ego-motion noise is now measured over each bystander's actual measurement window (cached per window so co-windowed bystanders pay Farneback once), and every output record carries `measurement_window_sec` + `window_source` provenance. **Validated on the June 10 re-run (`e2e_reports/2026_06_10_layer03d_50_postfix/`): 0/50 → 48/50 clips scored, 112 person-task vectors — every one via `bystander_anchored`, retro-confirming the geometry diagnosis — 10 confident vectors with physically consistent signs, and both directional exemplars visually verified** (`343f4d2d` Approach +0.41: pedestrian ahead of the wearer grows 7.8 % → 10.1 % of frame; `599f2f09` Avoidance −0.48: festival-goer shrinks 5.1 % → 1.0 %). The 2 remaining sentinels are legitimate (all bystanders single-detection). Covered by 5 new tests (window kept when dense, anchored when sparse, edge-clipped, single-detection `None`, end-to-end sparse scoring); suite 136/136.

2. **SAM Bbox-Prompt Crashed on MPS (float64) — Silent Raw-Bbox Fallback (Resolved - June 10)**:
   - **Problem**: On the Mac Studio (MPS), every `_segment_with_sam` call failed with `Cannot convert a MPS Tensor to float64 dtype` — `SamProcessor` emits float64 tensors (`input_boxes`/`original_sizes`) that `.to('mps')` cannot host — and the broad `except` silently fell back to the rectangular-bbox depth mask. The documented "Bbox-prompted SAM Instance Masking" therefore **never ran in production**: `facebook/sam-vit-huge` (~2.5 GB) loaded at init then failed on every frame, and depth medians were diluted by background pixels inside the box. Same silent-degradation pattern eliminated in 03b (mock emotions) and 03c (missing deps).
   - **Solution**: In `_segment_with_sam`, processor outputs are now cast float64 → float32 **before** moving to device (integer size tensors untouched), and a **one-time loud warning** (`self._sam_failure_warned`) fires if SAM inference ever fails, so a silent batch-wide fallback cannot recur. Validated single-frame: the mask produces a 146 k px person silhouette vs the 569 k px raw bbox, masked depth median **193 vs 72** (the bbox median had been averaging in dark background); the June 10 50-clip run log shows **0 SAM failures**.

3. **Anchored Measurement Span Was Unbounded on Sparse Tracks (Resolved - June 11)**:
   - **Problem**: Surfaced by the June 10 post-fix run: `_bystander_measurement_window()` widens to the climax-nearest detection ± 1 *detection index*, so the span in **seconds** is whatever the neighbor-detection gap happens to be — on that corpus **33/112 vectors sat on spans > 30 s (max 198 s)**. A bbox/depth delta measured over minutes of track gap is locomotion/drift, not a task reaction. The risk was latent in that run (all 33 long-span vectors were zeroed by the ego-motion chaos gate; all 10 confident vectors sat on ≤ 12 s spans), but a quiet-camera clip with a sparse track would have passed a minutes-long "reaction" vector at confidence 1.0, indistinguishable from a tight 6 s measurement.
   - **Solution** (Option A): Added `MAX_ANCHOR_SPAN_SEC = 30.0` to the tuning-constants block; `_bystander_measurement_window()` returns `None` (bystander skipped, sentinel semantics unchanged) when the anchored span exceeds it — the strict-`reaction_window` path is unaffected (2 s by construction). **Validated against the recorded June 10 run: exactly the 33 long-span vectors are now skipped (every one verified > 30 s), the remaining 79 windows are bit-identical, and all 10 confident vectors are retained untouched** — a provably surgical change. Covered by `test_measurement_window_caps_long_anchored_span` (200 s gap → `None`) and `test_measurement_window_at_cap_boundary_kept` (exactly 30 s → kept; the cap is strictly-greater-than); suite 138/138.

## ⚠️ Unresolved Issues & Suggestions

_None at this time — the June 9–11 smell-test findings (window/cadence geometry, SAM MPS float64, unbounded anchored span) have all been resolved above._
