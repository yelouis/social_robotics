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
- **Batch Test**: Run across a batch of standard interaction videos. Validate that `Depth Anything V2-Small` tensor offloading functions correctly on the **Mac Studio (M4 Max, 64 GB unified memory)** via the PyTorch MPS backend. Assert that the `proxemic_vector` appropriately penalizes jitter (ignoring +/- 0.05 micro-movements to avoid false positive "lunges"). If V2-Small accuracy proves insufficient on the new host's memory budget, V1-Large or Metric3D v2 are now viable upgrades (see Unresolved Issue 1) — both Apache-2.0.

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

   - **Problem**: Setting `os.environ['HF_HOME']` inside the class constructor was too late, as the `transformers` library initializes its cache paths upon module-level import.
   - **Solution**: Moved the `HF_HOME` environment assignment to the absolute top of `pipeline.py`, ensuring it precedes any `transformers` or `torch` imports.

6. **Micro-movement Aliasing via Adaptive Sampling (Resolved - May 05)**:
   - **Problem**: Randomly seeking up to 5 frames in large reaction windows (>2.0s) led to sparse temporal coverage (>1s gaps), which risked missing rapid approach or retreat events within the window.
   - **Solution**: Replaced the static 5-frame hardcap with a dynamic frame count proportional to the window duration `max(5, int(window_duration * 3))`, capping at 20 frames, thereby ensuring consistent 3 FPS coverage.

7. **Test Suite Resumability Guard & Side-Effect Leakage (Resolved - May 05)**:
   - **Problem**: The `test_schema_conformance` test applied a global patch to `pathlib.Path.exists` to mock video presence. This caused a global side-effect leak that bypassed the pipeline's natural resumability checks (`self.output_result_path.exists()`).
   - **Solution**: Removed the global `@patch` and instead created a legitimate empty dummy file `dummy.mp4` within the `tmp_path` fixture to cleanly satisfy the `.exists()` validation without interfering with other path assertions.

8. **Heuristic Signal Conflict via Optical Flow Noise Rejection (Resolved - May 05)**:
   - **Problem**: In scenarios with extreme camera panning, bounding box scale expansion (due to perspective distortion) falsely indicated an "approach", contradicting the depth map deltas and corrupting the `proxemic_vector`.
   - **Solution**: Implemented an `_extract_ego_motion_noise` validation pass utilizing Farneback optical flow. If the 95th percentile magnitude exceeds a noise threshold of `15.0`, the pipeline zeroes the `proxemic_vector` and flags `proxemic_confidence = 0.0` to prevent polluting downstream systems. Additionally, a dynamic `proxemic_confidence` is calculated based on the sign alignment between bounding box and depth delta heuristics.

9. **Missing Dependency Validation (Resolved - May 05)**:
   - **Problem**: The pipeline threw a generic error when encountering missing `transformers` or `torch` dependencies, lacking actionable instructions for environment resolution.
   - **Solution**: Replaced the generic `RuntimeError` with a specific installation directive containing the exact required command: `pip install transformers>=4.35.0 huggingface_hub torch`.

10. **Occlusion Glitches via SAM-1 Instance Masking (Resolved - May 05)**:
    - **Problem**: The pipeline masked the depth map using a rectangular YOLO `person` bounding box. When objects (e.g., hands, tools, furniture) occluded the bystander, their depth values polluted the bounding box median, causing false "approach" spikes.
    - **Solution**: Integrated the `facebook/sam-vit-base` SAM-1 model natively via the HuggingFace `mask-generation` pipeline. The pipeline now crops the bounding box, generates precise instance masks, selects the largest mask (representing the bystander), and uses this precise mask to filter the depth map, drastically reducing occlusion noise.

11. **SAM-1 Automatic Mask Generation Latency (Resolved - May 07)**:
    - **Problem**: SAM was wired through `transformers.pipeline(task="mask-generation", ...)`, which runs the auto mask generator (32×32 = 1024 candidate point prompts) over the cropped bystander region for *every* sampled frame. With 5-20 frames per task per bystander, SAM dominated the M4 Pro MPS wall-clock cost despite the documented intent ("crops the bounding box, generates precise instance masks, selects the largest mask") only requiring a single bbox-seeded mask.
    - **Solution**: Replaced the `mask-generation` pipeline with `SamModel.from_pretrained("facebook/sam-vit-base")` + `SamProcessor.from_pretrained(...)` and a new `_segment_with_sam(img, x1, y1, x2, y2)` helper. The bystander bbox is passed as `input_boxes=[[[x1, y1, x2, y2]]]`; one forward pass per frame instead of 1024. The best mask is selected by SAM's IoU score head (`outputs.iou_scores`) rather than by a largest-area heuristic that previously gambled on background masks. Output masks are at original-frame resolution and resized to the depth-map resolution via PIL nearest-neighbor.
    - **Model Instructions/Context**: When integrating SAM family models elsewhere in the codebase, prefer the lower-level `SamModel`/`SamProcessor` API with prompt-based inputs (`input_boxes` or `input_points`) over the high-level `mask-generation` pipeline whenever the prompt geometry is already known. The IoU-score head is the canonical mask selector — never rely on "largest mask" heuristics in scenes with complex backgrounds.

12. **Two-Endpoint Depth Delta Discards Intermediate Samples (Resolved - May 07)**:
    - **Problem**: `_calculate_depth_delta` only consumed `depths[0][1]` and `depths[-1][1]`, so any noise, occlusion glitch, or SAM mask error on the first or last frame propagated directly into the `proxemic_vector` while the 3-18 intermediate samples collected at significant compute cost were discarded. This partially undermined the rationale for the adaptive 3 FPS sampling (Resolved Issue #6, May 5).
    - **Solution**: Introduced a static `_slope_span_delta(depths)` helper that fits a least-squares line through `(t, median_depth)` pairs via `np.polyfit(ts, ds, 1)` and returns `slope * window_duration` — a unit-consistent "predicted span" that uses all collected samples. The scale matches the previous endpoint-delta convention for linear data, so the existing `-depth_delta * 2.0` calibration is preserved. Added `test_slope_span_robust_to_endpoint_outlier` (asserts -0.4 vs -0.5 endpoint delta on outlier-dominated input) and `test_slope_span_matches_endpoint_for_linear_data` (asserts equality on noiseless input).
    - **Model Instructions/Context**: Time-series deltas in this codebase should prefer slope-span over endpoint subtraction whenever multiple intermediate samples are available. The `slope * window_duration` formulation preserves dimensional consistency with the original endpoint scale and recalibrating downstream constants is rarely required.

13. **Failed/Empty Videos Reprocessed on Every Resume (Resolved - May 07)**:
    - **Problem**: When `process_video` returned `None` (missing file, no bystanders, no tasks, or all tasks filtered), the `if result:` guard skipped `processed_ids.add(...)`. Every resume re-iterated these videos, re-invoking `_extract_ego_motion_noise` (full optical-flow scan) and `_calculate_depth_delta` (Depth Anything + SAM per frame), and discarded the result again.
    - **Solution**: The `run()` loop now appends a sentinel record `{"video_id": ..., "layer": "03d_proxemic_kinematics", "tasks_analyzed": [], "skipped_reason": "no_output_produced"}` whenever `process_video` returns `None`, and adds the video_id to `processed_ids`. Resume cost on these videos drops to O(JSON-load). Errors caught by the outer `except` are intentionally still un-marked so transient failures (network, OOM, GPU resets) get retried on the next run. Added `test_sentinel_record_for_missing_video` to lock in the persistence contract.
    - **Model Instructions/Context**: Downstream consumers of `03d_proxemic_kinematics_result.json` must filter records by `len(tasks_analyzed) > 0` or by `skipped_reason in entry`. Sentinel records are a deliberate part of the schema, not a bug. Apply this skip-record pattern to any future resumable layer that performs expensive per-video work.

14. **Hardcoded SSD Cache Path with No Mount Validation (Resolved - May 07)**:
    - **Problem**: `SSD_HF_CACHE = "/Volumes/Extreme SSD/huggingface_cache"` was set before transformers import, but `os.makedirs(...)` succeeded silently when the SSD was not mounted — macOS does not block writes under `/Volumes/<name>` when nothing is actually mounted there. Result: ~500MB of model weights spilled onto the boot disk, the exact failure mode the SSD cache was introduced to prevent. The previous post-init check only emitted a print warning.
    - **Solution**: Added `if not os.path.ismount(ssd_root): raise RuntimeError(...)` at the very top of `_init_model`, before `os.makedirs` and any model construction. The error message is actionable: it states which volume is missing, what the layer needs the cache for, and how to override `SSD_HF_CACHE` for hosts without the SSD. Added `test_init_raises_when_ssd_not_mounted` to enforce the fail-fast contract.
    - **Model Instructions/Context**: For any future layer that pins a cache or output path under `/Volumes/<name>`, always gate construction on `os.path.ismount(...)` rather than trusting `os.path.exists(...)` or `os.makedirs(..., exist_ok=True)`. A phantom directory on the boot volume is silent disk-pressure waiting to happen.

15. **GPU/MPS Memory Not Released Across Long Batch Runs (Resolved - May 07)**:
    - **Problem**: Both Depth Anything V2-Small and SAM ViT-Base were instantiated once and held for the pipeline lifetime. Neither `torch.mps.empty_cache()` (M4 Pro) nor `torch.cuda.empty_cache()` was called between videos, so MPS unified-memory pressure accumulated from cached intermediate activations across batches with 10-20 frame samples × multiple bystanders × dozens of videos — risking allocation failures or thermal throttling on the 24GB shared budget.
    - **Solution**: Added `_release_accelerator_cache()` and call it from a `finally` block in the `run()` per-video loop. The helper guards on `self.device` and `hasattr(torch, 'mps')` so it's a no-op on CPU and survives older PyTorch builds without `torch.mps`. Cache-flush exceptions are caught and swallowed (best-effort) so a broken backend cannot break the run loop. Added `test_release_accelerator_cache_noop_on_cpu` to enforce the CPU-safe contract.
    - **Model Instructions/Context**: Any layer using GPU/MPS-resident transformer weights across a long video batch should follow the same `finally: self._release_accelerator_cache()` pattern in its run loop. The flush is fast (~milliseconds) and the steady-state memory savings compound across concurrent layers (03a + 03b + 03c + 03d) on the 24GB Mac mini.

16. **Hardcoded Heuristic Constants Throughout the Vector Pipeline (Resolved - May 07)**:
    - **Problem**: Magic numbers governing proxemic classification were inlined throughout `pipeline.py` (`noise_threshold = 15.0`, `bbox_delta / 50.0`, `-depth_delta * 2.0`, weights `0.4`/`0.6`, deadband `0.05`, action thresholds `±0.3`). None were exposed via constructor arguments, configuration, or class-level constants — retuning required hunting through helpers.
    - **Solution**: Hoisted all seven constants into a `# --- Tuning Constants ---` block at the top of `ProxemicKinematicsPipeline` (`OPTICAL_FLOW_NOISE_THRESHOLD`, `BBOX_NORM_PCT`, `DEPTH_NORM_SCALE`, `BBOX_WEIGHT`, `DEPTH_WEIGHT`, `MICROMOVEMENT_THRESHOLD`, `APPROACH_THRESHOLD`). All inline literals in `process_video` and `_compute_proxemic_vector` now reference `self.<NAME>`. Subclassing for ablations is a one-line override. Added `test_tuning_constants_subclass_override` to verify that lowering `APPROACH_THRESHOLD` reclassifies a borderline vector without source edits.
    - **Model Instructions/Context**: When tuning thresholds across runs (e.g., adult vs. infant scenes, switching to V1-Large), subclass `ProxemicKinematicsPipeline` and override the relevant class-level constant rather than editing source. This pattern matches the Layer03cConfig dataclass pattern adopted in 03c — both centralize the tuning surface; subclass-of-pipeline is preferred here because there are no per-instance configuration variants on the manifest path.

17. **Model Capability Bottleneck on 64GB Hosts (Resolved - May 14)**:
    - **Problem**: Depth Anything V2-Small was pinned to accommodate the 24GB memory limit of previous Mac mini hosts, sacrificing edge fidelity on bystander silhouettes.
    - **Solution**: Upgraded the model unconditionally to `LiheYoung/depth-anything-large-hf` (Depth Anything V1-Large) to leverage the 64GB unified memory of the Mac Studio.

18. **SAM ViT-Base Occlusion Accuracy Headroom (Resolved - May 14)**:
    - **Problem**: The pipeline used `facebook/sam-vit-base`, which struggled with sharply masking heavily occluded bystanders compared to larger parameter variants.
    - **Solution**: Swapped the model to `facebook/sam-vit-huge` as a drop-in replacement, improving silhouette precision at the cost of expected inference latency well within the M4 Max budget.

19. **Per-Video Cache Flush Overhead (Resolved - May 14)**:
    - **Problem**: `_release_accelerator_cache()` was invoked after every single video, which defended against OOM on 24GB hosts but wasted wall-clock time on 64GB hosts that can safely buffer many videos worth of activations.
    - **Solution**: Introduced an auto-tuning `FLUSH_EVERY_N_VIDEOS` class constant. It dynamically evaluates host memory via `os.sysconf` at import time, setting the flush interval to `25` on hosts with >= 48GB memory and falling back to `1` on smaller hosts.

20. **Depth and SAM Identifiers Migrated to Tier-Per-Host Registry (Resolved - May 16)**:
    - **Problem**: After Resolved Issues #17 (Depth Anything V1-Large promotion) and #18 (SAM ViT-Huge promotion), both model identifiers were hardcoded string literals in `_init_model` (`pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", ...)` and `SamModel.from_pretrained("facebook/sam-vit-huge")`). The HF-cache-presence-check path (`model_cache_path = Path(SSD_HF_CACHE) / "hub" / "models--LiheYoung--depth-anything-large-hf"`) also baked in the slug. The pre-Mac-Studio "small" variants (`Depth-Anything-V2-Small-hf` + `sam-vit-base`) were therefore unreachable without source edits, and the `RuntimeError` raised on init failure still mentioned "Depth Anything V2" even though V1-Large was actually loading.
    - **Solution**: `pipeline.py` now imports `get_model` from `src/models_config.py` and resolves `depth_model_id = get_model("layer_03d_depth")` + `sam_model_id = get_model("layer_03d_sam")` once per pipeline construction. The depth `pipeline(...)`, both `SamModel.from_pretrained(...)` / `SamProcessor.from_pretrained(...)` calls, the init log messages, and the HF-cache-presence-check slug (derived as `"models--" + depth_model_id.replace("/", "--")`) all read from those locals. The error message was retargeted from "Failed to initialize Depth Anything V2" to "Failed to initialize depth/SAM stack" so it stays accurate across all tiers. The Extreme-SSD-mount precondition error message was also broadened from "Depth Anything V2 + SAM weights (~500MB)" to "depth + SAM weights (~0.5-4 GB depending on tier)" to reflect the per-tier footprint range. `medium`/`large` preserve the V1-Large + ViT-Huge production defaults; `small` substitutes Depth Anything V2-Small + SAM ViT-Base for the 24 GB Mac mini fallback. See `ml_dependencies.md` Resolved Issue #1 for the cross-layer registry rationale.

## ⚠️ Unresolved Issues & Suggestions
*None*