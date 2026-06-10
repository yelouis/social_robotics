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

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Reaction-window vs detection-cadence geometry guarantees 0 yield (0/50 scored)
**Status**: ⚠️ Confirmed Unresolved — The **June 9 50-clip smell test** (`e2e_reports/2026_06_09_layer03d_50/`) scored **0/50 clips: every record is a `no_output_produced` sentinel** (run clean, no errors, 181 s total). Root cause verified in the manifest math: `_calculate_bbox_scale_delta` and `_calculate_depth_delta` (`src/layer_03d_proxemic_kinematics/pipeline.py`) require **≥ 2 bystander detections strictly inside `task_reaction_window_sec`**, but Node-02 bystander detections arrive at a **median 6.0 s cadence** (1/3 FPS sampling, sparser after filtering) while the reaction windows are **2.0 s wide** — a window geometrically holds **at most one** detection, so ≥ 2 is impossible by construction: **0/59 tasks** had ≥ 2 in-window detections. Anchoring compounds it: the window is anchored to the *wearer's* optical-flow climax, and the nearest bystander detection is a **median 6.8 s (max 158 s)** from the window center — the same wearer-vs-bystander mismatch 03b fixed in its Resolved #2. Mechanics were proven sound by a probe that widened windows to span a contiguous detection run: **4/4 clips produced real proxemic vectors** (e.g. `044a7a23` bbox +26.7 %, vector +0.21), and the depth spot-check (`frames/depth_spotcheck_*.jpg`) shows Depth-Anything cleanly resolving the bystander with the median tracking proximity (0.13 → 0.28 → 0.12 as they approach then recede).

**Option A (recommended)**: **Per-bystander window anchoring + detection-span widening.** Anchor each bystander's measurement interval to its detections nearest the climax (the proven 03b Resolved #2 pattern, ideally extracted into a shared helper), and widen the effective sampling span to the nearest **N ≥ 2 consecutive detections** around the climax (e.g. ±1–2 detections, ~6–12 s), since any 2 s window is single-detection by construction at the Node-02 cadence.
  - *Pros*: No new inference; mirrors the validated 03b fix; the probe already demonstrated 4/4 mechanics on exactly this pattern; shared-helper refactor benefits every window-consuming layer.
  - *Cons*: The measured interval extends beyond the strict reaction window, so the vector may include non-reaction locomotion; the approach/avoid semantics need re-validation on real clips.

**Option B**: **Re-detect bystanders inside the window at ~3 FPS** with YOLO-pose (the 03a Resolved #2 pattern), instead of consuming only the sparse manifest detections — fulfilling the doc's "adaptive sampling at ~3 FPS" intent with fresh boxes.
  - *Pros*: Measures the true reaction window at the documented density; fresh boxes (no multi-second-stale drift); YOLO-pose is cheap next to the depth+SAM stack already loaded.
  - *Cons*: Adds per-frame inference + an IoU identity-match step between re-detections and manifest `person_id`s; still needs the 03b-style anchor for clips where the bystander is entirely off-camera during the wearer's climax.

**Option C**: **Consume 03a's `attention_trace` as a cross-layer box source** (8–32 FPS positions per bystander) when `03a_attention_result.json` is available.
  - *Pros*: Zero new inference; densest available signal.
  - *Cons*: Introduces an ordering dependency on 03a (currently "Cross-layer: None"); 03a traces only cover 03a's own sampled windows, so coverage is not guaranteed.

Your selection: _____

---

### Issue 2: SAM bbox-prompt crashes on MPS (float64) — production silently falls back to raw-bbox depth medians
**Status**: ⚠️ Confirmed Unresolved — Observed live on every probed frame during the June 9 smell test: `_segment_with_sam` logs `SAM bbox-prompt inference failed: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.` The `SamProcessor` emits float64 tensors (`input_boxes` / `original_sizes`) which `.to('mps')` cannot host; the broad `except` then returns `None` and `_calculate_depth_delta` **silently** substitutes the rectangular-bbox mask. Net effect on the Mac Studio (MPS): the documented "Bbox-prompted SAM Instance Masking" **never runs** — `facebook/sam-vit-huge` (~2.5 GB) is loaded at init and then fails on every frame — and depth medians are diluted by background pixels inside the box (visible in `frames/depth_spotcheck_2_t237.jpg`, where the box spans a dark doorway behind the bystander). This is the silent-degradation pattern the project has eliminated elsewhere (03b mock emotions, 03c missing deps).
**Remediation (single option — straightforward dtype fix)**: Cast the processor outputs to MPS-compatible dtypes before moving to device (e.g. coerce any `float64` tensor in `inputs` to `float32`; keep integer tensors as-is), and add a one-time loud warning if SAM inference fails so a silent fallback can never run an entire batch unnoticed. Validate on one clip that the SAM mask path produces a mask and that the masked median differs from the bbox median.
  - *Pros*: Restores the documented instance-masking precision on the primary host; removes a silent 2.5 GB dead-weight load; trivially testable.
  - *Cons*: None of substance — pure dtype-compatibility fix; masked medians will shift slightly vs the (incorrect) bbox medians, which is the intended correction.

Your selection: _____
