# AI Task Breakdown: Proxemic Kinematics Layer (03d)

## Objective
The **Proxemic Kinematics Layer** investigates "Approach vs. Avoidance." In developmental stages, an infant assesses the severity of an action by how the caregiver physically manages space (Proxemics). If a child touches something dangerous, the caregiver lunges forward (Approach/Intervention). If they smell something rotten, the caregiver recoils (Avoidance). This layer measures the physical vector of the bystander relative to the POV camera.

---

## 📥 Input Requirements
- **`filtered_manifest.json`**: Needs the `bystander_detections` array containing the bounding box coordinates of the other person across time, and the `task_reaction_window_sec`.
- **Cross-layer (optional)**: None strictly required, though it correlates heavily with motor resonance.

---

## 🛠️ Implementation Strategy

**Measurement window (bystander-anchored).** Both methods below measure a *delta over time*, so each needs ≥ 2 bystander detections inside the measurement interval. Node-02 emits bystander detections at a ~6 s median cadence while `task_reaction_window_sec` is only ~2 s wide, so a strict reaction window holds at most one detection and would yield **nothing by construction**. The interval is therefore computed by `_bystander_measurement_window` (the shared `src/shared/bystander_window.py` helper — see docs/03 Cross-Layer § Shared Helper): the strict reaction window is kept when it already holds ≥ 2 detections, otherwise it re-anchors to the bystander's climax-nearest detection ± 1 detection — **capped at `MAX_ANCHOR_SPAN_SEC = 30 s`** so a sparse track cannot read minutes of locomotion/drift as a "reaction." Every record carries `measurement_window_sec` + `window_source` provenance. *(Rationale and validation: the geometry mismatch that made ≥ 2 detections impossible is Resolved #1; the span cap that prevents minutes-long false vectors is Resolved #3.)* Each task also expands into one segment per bystander cluster (multi-window — docs/03 § Multi-Window Reaction Segments), so 03d applies the cross-layer per-segment guardrails: **skip untracked (negative-id) bystanders**, and **dedupe** segments that re-anchor to the same `(person, measurement_window)` — otherwise a phantom track is counted once per segment.

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
   - **Solution** (Option A): Added `_bystander_measurement_window()` (`src/layer_03d_proxemic_kinematics/pipeline.py`; **June 14: extracted to the shared `src/shared/bystander_window.py` helper — see docs/03 Cross-Layer § Shared Helper**): the strict reaction window is kept when it already holds ≥ 2 detections (dense-detection behavior unchanged); otherwise the measurement interval re-anchors to the bystander's climax-nearest detection ± `ANCHOR_SPAN_DETECTIONS` (= 1), a 2–3-consecutive-detection span (~6–12 s nominal). Ego-motion noise is now measured over each bystander's actual measurement window (cached per window so co-windowed bystanders pay Farneback once), and every output record carries `measurement_window_sec` + `window_source` provenance. **Validated on the June 10 re-run (`e2e_reports/2026_06_10_layer03d_50_postfix/`): 0/50 → 48/50 clips scored, 112 person-task vectors — every one via `bystander_anchored`, retro-confirming the geometry diagnosis — 10 confident vectors with physically consistent signs, and both directional exemplars visually verified** (`343f4d2d` Approach +0.41: pedestrian ahead of the wearer grows 7.8 % → 10.1 % of frame; `599f2f09` Avoidance −0.48: festival-goer shrinks 5.1 % → 1.0 %). The 2 remaining sentinels are legitimate (all bystanders single-detection). Covered by 5 new tests (window kept when dense, anchored when sparse, edge-clipped, single-detection `None`, end-to-end sparse scoring); suite 136/136.

2. **SAM Bbox-Prompt Crashed on MPS (float64) — Silent Raw-Bbox Fallback (Resolved - June 10)**:
   - **Problem**: On the Mac Studio (MPS), every `_segment_with_sam` call failed with `Cannot convert a MPS Tensor to float64 dtype` — `SamProcessor` emits float64 tensors (`input_boxes`/`original_sizes`) that `.to('mps')` cannot host — and the broad `except` silently fell back to the rectangular-bbox depth mask. The documented "Bbox-prompted SAM Instance Masking" therefore **never ran in production**: `facebook/sam-vit-huge` (~2.5 GB) loaded at init then failed on every frame, and depth medians were diluted by background pixels inside the box. Same silent-degradation pattern eliminated in 03b (mock emotions) and 03c (missing deps).
   - **Solution**: In `_segment_with_sam`, processor outputs are now cast float64 → float32 **before** moving to device (integer size tensors untouched), and a **one-time loud warning** (`self._sam_failure_warned`) fires if SAM inference ever fails, so a silent batch-wide fallback cannot recur. Validated single-frame: the mask produces a 146 k px person silhouette vs the 569 k px raw bbox, masked depth median **193 vs 72** (the bbox median had been averaging in dark background); the June 10 50-clip run log shows **0 SAM failures**.

3. **Anchored Measurement Span Was Unbounded on Sparse Tracks (Resolved - June 11)**:
   - **Problem**: Surfaced by the June 10 post-fix run: `_bystander_measurement_window()` widens to the climax-nearest detection ± 1 *detection index*, so the span in **seconds** is whatever the neighbor-detection gap happens to be — on that corpus **33/112 vectors sat on spans > 30 s (max 198 s)**. A bbox/depth delta measured over minutes of track gap is locomotion/drift, not a task reaction. The risk was latent in that run (all 33 long-span vectors were zeroed by the ego-motion chaos gate; all 10 confident vectors sat on ≤ 12 s spans), but a quiet-camera clip with a sparse track would have passed a minutes-long "reaction" vector at confidence 1.0, indistinguishable from a tight 6 s measurement.
   - **Solution** (Option A): Added `MAX_ANCHOR_SPAN_SEC = 30.0` to the tuning-constants block; `_bystander_measurement_window()` returns `None` (bystander skipped, sentinel semantics unchanged) when the anchored span exceeds it — the strict-`reaction_window` path is unaffected (2 s by construction). **Validated against the recorded June 10 run: exactly the 33 long-span vectors are now skipped (every one verified > 30 s), the remaining 79 windows are bit-identical, and all 10 confident vectors are retained untouched** — a provably surgical change. Covered by `test_measurement_window_caps_long_anchored_span` (200 s gap → `None`) and `test_measurement_window_at_cap_boundary_kept` (exactly 30 s → kept; the cap is strictly-greater-than); suite 138/138.

4. **Untracked-Detection Fallback Collided with the Track-ID Namespace — Confident False Proxemic Vectors (Resolved - June 13)**:
   - **Problem**: The June 13 50-clip smell test's strongest Avoidance vector (`599f2f09` `person_id=2`, `proxemic_vector = −0.48`, `proxemic_confidence = 1.0`) was a confident false recoil. A faithful repro at the real 1/3 fps Node-02 sampling showed `person_id=2`'s track was `[(0.0, id2), (9.0, id2), (21.0, FALLBACK)]`: ByteTrack correctly tracked the banner-holder as id 2 at t=0/9, but the *different, distant child* at t=21 came back untracked (`box.id is None`), and `src/shared/social_presence.py` assigned it `person_id = len(frame_detections) = 2` — colliding with the banner-holder's real track id in the **same integer namespace**. So the apparent "ID-switch" was a **fallback-index collision, not a tracker mis-association**; an A/B confirmed ByteTrack, BoT-SORT (`gmc=sparseOptFlow`), and BoT-SORT + ReID all leave the child untracked at 1/3 fps, so the originally-considered tracker swap would not have fixed it and was withdrawn. 03d is most exposed (its signal is a temporal delta of one `person_id`'s box), but every `person_id`-consuming layer (03a/03b/03f) shared the upstream defect.
   - **Solution** (Option A + Option C): **Upstream root cause** — `social_presence.py` now assigns untracked detections (`box.id is None`) a unique monotonically-decreasing **negative** id (disjoint from ByteTrack's positive namespace) instead of `len(frame_detections)`, so the t=21 child becomes its own single-detection person and `person_id=2` reduces to the banner-holder alone (validated on the exemplar: child → id −7, `person_id=2` track = `[t=0, t=9]`). **Downstream defense (also repairs already-generated manifests)** — 03d adds an identity-continuity guard: `_window_identity_continuity` flags an in-window consecutive detection pair whose IoU ≤ `IDENTITY_IOU_FLOOR` (0.1) **and** area ratio falls outside `[IDENTITY_AREA_LO, IDENTITY_AREA_HI]` (= [0.6, 1.6]); a chaos-surviving flagged vector is zeroed with `identity_discontinuity: True` + `min_consecutive_iou` provenance. The guard sits **after** the ego-motion chaos gate, so chaos attribution is preserved and it targets exactly the confident false positives that survive chaos; the benign camera-pan case (IoU ≈ 0 but area ratio ≈ 1.0, one person displaced by a pan) is intentionally spared. **Validated on the June 13 50-clip re-run (`e2e_reports/2026_06_13_layer03d_50_postfix/`): Avoidance actions 1 → 0 (the false −0.48 zeroed), Approach 1 → 1 (the genuine `343f4d2d` +0.41 retained, min IoU 0.524), exactly 3 identity-discontinuity rejections — all genuine collisions (`599f2f09` pid 1 & 2, `18323b66`) — and all 13 genuine non-zero vectors plus both benign pans preserved; the 60/79 chaos attribution is unchanged.** Covered by 6 new tests (guard flags collision / spares pan / spares smooth track / single-box safe / end-to-end zeroing; negative-id fallback in `test_social_presence.py`); suite 157/157.

5. **Sentinel Records Were Opaque on the 30% No-Yield Corpus (Resolved - June 13)**:
   - **Problem**: The June 13 re-run scored 35/50 with 15 sentinels (vs June 10's 2 — the now-live span cap converting span-capped clips to no-yield), but every sentinel emitted the generic `no_output_produced` marker, so an operator could not tell *why* a clip yielded nothing (all bystanders span-capped vs single-detection vs absent looked identical). The originally-selected climax-proximity precondition was disproven and withdrawn: genuine confident vectors sit **43–68 s** from the wearer's optical-flow climax (bystander-anchoring, Resolved #1, intentionally decouples from it), so no `MAX_CLIMAX_OFFSET_SEC` threshold separates scoring vectors from sentinels without deleting real signal — including the layer's best genuine vector.
   - **Solution** (Option A): `_bystander_measurement_window` now returns `(None, None, reason)` (`"single_detection"` or `"span_capped"`) instead of a bare `None`; `process_video` aggregates the per-bystander reasons into a specific sentinel `skipped_reason` (`all_bystanders_span_capped`, `all_bystanders_single_detection`, `mixed_skip`, `no_bystanders`, `no_tasks`). Pure labeling — the scoring path is byte-for-byte unchanged. **Validated on the June 13 re-run: the 15 sentinels now read 7 `all_bystanders_span_capped`, 6 `mixed_skip`, 2 `all_bystanders_single_detection`** — the 30% no-yield rate is self-explanatory in the output JSON. Covered by 4 new sentinel-reason tests; suite 157/157.

6. **Bystander Track-Explosion Made the Full Run Infeasible — Per-Clip Cap (Resolved - June 27)**:
   - **Problem**: The first full top-200 run attempt was infeasible. Node-02 fragments bystander tracking into many short spurious **positive-id** tracks — median **48**, up to **829 "persons"/clip** (median 4 detections each), **18,497 depth+SAM tracks corpus-wide**. 03d pays Depth-Anything-Large + SAM-Huge per genuine bystander (several frames each), so this projected to a **multi-day** run — and it is wrong to the *motive*: 829 "bystanders" in one clip are ByteTrack fragments, not real people. The negative-id untracked filter (the cross-layer multi-window guardrail) does **not** catch them — they carry positive ids.
   - **Solution**: `MAX_BYSTANDERS_PER_CLIP = 10` — process only the N longest tracks per clip (the genuine, sustained bystanders a scene actually has); sorting by detection count puts real bystanders first and drops short fragments. Bounds the run to ~1,945 depth-tracks (clip `46bcee63`: 732 → 10). **Validated** on the full top-200: **200 clips, 0 errors, max 10 persons/clip**, 108 Approach / 77 Avoidance (185 non-neutral vectors), **~14.8 h** (vs days). The same cap is carried into 03f (per-bystander YOLO-pose). The heavy per-bystander layers (03d/03f) also run via the shared **parallel harness** (`tools/run_parallel_layer.py`) for **~3×** throughput — stress-tested at N=3 on the heaviest clips with no crash and 80% RAM free.

7. **Sparse Boxes Suppressed Scorability — Window-Dense Pre-Pass + Additive Proxemic Trajectory (Resolved - June 30)**:
   - **Problem**: 03d's proxemic delta needs **≥ 2 endpoints** in the measurement window; at Node-02's 1/3-fps sampling many clips skipped as `mixed_skip` / `all_bystanders_single_detection`. A 25-clip A/B (`tools/ab_density.py`, docs/02 Issue 1) feeding window-dense boxes recovered scorability — 22 → 25 clips with data, +24 per-person rows, `proxemic_confidence>0` 7 → 11, and 0 → 2 genuine non-Neutral actions — a **moderate** but real gain (most verdicts stay `Neutral` because a 2 s delta is physically often neutral).
   - **Solution** (Option A): Adopt the window-dense manifest (`tools/densify_manifest.py` over `src/shared/dense_detect.py`; **no 03d classifier change**, like 03f) **and** add an **additive per-frame proxemic trajectory** — `_compute_proxemic_trajectory` characterizes the in-window bbox-scale series as `proxemic_trajectory_shape` ∈ `TRAJECTORY_SHAPES` {`insufficient`/`flat`/`monotonic_approach`/`monotonic_retreat`/`oscillatory`} + a downsampled `proxemic_trajectory_pct`, leaving the existing 2-endpoint delta/`classified_action` untouched. Surfaced as queryable Layer-04 columns (`any_approach_trajectory` / `any_retreat_trajectory` / `any_oscillatory_trajectory` via `LAYER_SUMMARY_REGISTRY`, pinned in the enum-sync test). **Landed on the top-200** (`tools/run_density_landing.py`, ~14.8 h via the parallel harness): non-Neutral **185 → 207** (high-confidence `conf>0` 128 → 143), one clip pruned (200 → **199** rows, longest-track-sort reorder), and **re-published** to `louisye/social-robotics-proxemic-kinematics` (199 rows × 13 cols incl. the trajectory columns). Spot-checked via the Layer-05 visualizer — the `0b530687` `monotonic_approach` matched the video, and its `proxemic_confidence: 0.0` correctly flagged the camera-vs-subject ambiguity. **Consumer guidance**: filter approach/avoidance by `proxemic_confidence`.

## ⚠️ Unresolved Issues & Suggestions

_No actionable issues — the June 13 smell-test findings (Resolved #4–#5), the June 27 track-explosion infeasibility (Resolved #6), and the June 30 dense-box adoption + trajectory (Resolved #7) are resolved above._

_Observation (not yet a formal issue): bystander-anchoring (Resolved #1) measures some genuine vectors **40–70 s from the wearer's task climax** (e.g. the `343f4d2d` Approach at 67.6 s), which raises a fidelity question — is a proxemic delta that far from the climax really a task reaction? Recorded here for visibility; open it as a formal issue if tightening task-reaction locality becomes a priority._
