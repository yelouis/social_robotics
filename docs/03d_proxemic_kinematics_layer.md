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

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: SAM-1 Automatic Mask Generation Latency
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:69` and `pipeline.py:372`. The pipeline initializes SAM via `transformers.pipeline(task="mask-generation", ...)`, which runs HuggingFace's automatic mask generator (default 32x32 = 1024 candidate point prompts) over the cropped bystander region for *every* sampled frame. With 5-20 frames per task per bystander, the wall-clock cost on the Mac mini M4 Pro (MPS backend) is dominated by SAM rather than Depth Anything. The documented intent is "crops the bounding box, generates precise instance masks, selects the largest mask," which only requires a *single* mask seeded by the bbox itself, not exhaustive automatic discovery.

**Option A (recommended)**: **Bbox-Prompted SAM via `SamModel` + `SamProcessor`** — Replace the `mask-generation` pipeline with the lower-level `SamModel`/`SamProcessor` API and pass `input_boxes=[[[0, 0, w, h]]]` (or the bystander bbox in original frame coordinates). This produces one mask seeded by the bbox prompt and aligns with the documented architecture.
  - *Pros*: 10-50x speedup per frame (one forward pass vs. 1024 grid points); deterministic mask selection (no "largest area" heuristic gambling on background masks); matches documented design intent.
  - *Cons*: Requires refactor away from the high-level `pipeline()` abstraction; manual handling of `pixel_values`, `input_boxes`, and post-processing tensors; need to manage device placement explicitly.

**Option B**: **Reduce Mask-Generation Grid Density** — Pass `points_per_side=8` (or similar) to the existing `mask-generation` pipeline to reduce the candidate point grid from 32x32 to 8x8.
  - *Pros*: Minimal code change; preserves current call structure.
  - *Cons*: Still 16x slower than bbox prompting; quality of "largest mask" selection degrades with fewer candidates; does not address the architectural mismatch.

**Option C**: **Cache SAM Masks Per (video_id, frame_idx, bbox)** — Persist generated masks to disk keyed by video and frame to amortize cost across re-runs.
  - *Pros*: Eliminates SAM cost on resume/re-runs; useful for ablation experiments.
  - *Cons*: Disk I/O overhead and cache management; does not help the first run; cache invalidation complexity if bbox detector upstream is retuned.

Your selection: Proceed with Option A.

---

### Issue 2: Two-Endpoint Depth Delta Discards Intermediate Samples
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:413-416`. Despite sampling up to 20 depth values across the reaction window (per Issue 6's adaptive sampling fix), `_calculate_depth_delta` only consumes `depths[0][1]` and `depths[-1][1]`. Any noise, occlusion glitch, or SAM mask error on the first or last frame propagates directly into the `proxemic_vector`, while the intermediate 3-18 samples — collected at significant compute cost — are discarded. This partially undermines the rationale for adaptive sampling, which was meant to ensure consistent 3 FPS coverage to detect rapid events within the window.

**Option A (recommended)**: **Linear Regression Slope** — Fit a least-squares line through `(t, median_depth)` pairs and use the slope (Δdepth/Δt) as the delta signal. Normalizing by window duration yields a unit-consistent rate-of-approach.
  - *Pros*: Robust to single-frame outliers at endpoints; uses all collected samples; physically meaningful (rate of approach in normalized depth/sec).
  - *Cons*: Slightly more complex normalization in `_compute_proxemic_vector`; requires recalibration of the `-depth_delta * 2.0` scaling constant.

**Option B**: **Median of First-Third vs. Last-Third** — Compute medians over the first and last thirds of the sample list, then take the difference.
  - *Pros*: Simple change; preserves "start vs. end" semantics while resisting endpoint outliers; no calibration shift required.
  - *Cons*: Still discards the middle third of samples; degrades to two-point delta for windows yielding only 5 samples.

**Option C**: **Trimmed-Mean Endpoints** — Use the mean of the first 2-3 samples and the mean of the last 2-3 samples for the delta.
  - *Pros*: Minimal logic change; smooths endpoint noise.
  - *Cons*: Fixed window-size assumption; sensitive to systematic drift in the first/last samples.

Your selection: Proceed with Option A.

---

### Issue 3: Failed/Empty Videos Reprocessed on Every Resume
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:120-132`. When `process_video` returns `None` (file missing, no bystanders, no tasks, or all `tasks_analyzed` filtered out), the `if result:` guard skips both `results.append(...)` and `self.processed_ids.add(video_id)`. Consequently, every subsequent resume run re-iterates these videos, re-invokes `_extract_ego_motion_noise` (full optical-flow scan) and `_calculate_depth_delta` (Depth Anything + SAM per frame), and discards the result again. For batches with many edge-case videos, this consumes substantial time on guaranteed no-op work. Errors caught by the `except` branch (line 131) similarly never mark `video_id` processed.

**Option A (recommended)**: **Sentinel-Record Tracking** — When `process_video` returns `None`, write a sentinel record (e.g., `{"video_id": ..., "layer": "03d_proxemic_kinematics", "tasks_analyzed": [], "skipped_reason": "no_bystanders"}`) to results and mark the id processed. Filter sentinel records during downstream consumption.
  - *Pros*: Persists skip decisions; downstream layers gain explicit visibility into why a video was excluded; resume cost drops to O(JSON-load).
  - *Cons*: Inflates the result JSON with empty-task entries; downstream consumers must filter on `tasks_analyzed` length or `skipped_reason`.

**Option B**: **Separate Skip Manifest** — Maintain a sibling `03d_skipped.json` listing video_ids that produced no output, and short-circuit them at the top of the per-entry loop.
  - *Pros*: Keeps the main result JSON clean (only positive results); explicit skip log is easy to audit.
  - *Cons*: Two-file state machine to maintain; risk of skip-manifest/result-manifest drift if writes aren't atomic.

**Option C**: **Always-Mark-Processed Policy** — Always add `video_id` to `processed_ids` after `process_video` returns (regardless of value), and persist `processed_ids` to disk separately (e.g., `03d_processed_ids.json`).
  - *Pros*: Simplest write logic; cleanly decouples "did we attempt this?" from "did it produce output?"
  - *Cons*: Adds a third state file; tests must mock or write the new file; loses the rationale for *why* a video was skipped.

Your selection: Proceed with Option A.

---

### Issue 4: Hardcoded SSD Cache Path with No Mount Validation
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:6-7` and `pipeline.py:51`. `SSD_HF_CACHE = "/Volumes/Extreme SSD/huggingface_cache"` is set as a module-level constant before `transformers` import. If the external SSD is unmounted on the M4 Pro, `os.makedirs(SSD_HF_CACHE, exist_ok=True)` will succeed by creating a phantom `/Volumes/Extreme SSD/huggingface_cache` directory on the *root* filesystem (macOS does not auto-prevent writes under `/Volumes/<name>` when no volume is mounted at that name), causing model weights to spill onto the internal drive — the exact failure mode the SSD cache was introduced to prevent. The post-init existence check at `pipeline.py:72-74` only emits a print warning and does not abort.

**Option A (recommended)**: **Mount Validation with Hard Failure** — Before `os.makedirs`, verify `/Volumes/Extreme SSD` is an actual mount point via `os.path.ismount("/Volumes/Extreme SSD")` and raise `RuntimeError` with an actionable message if not mounted.
  - *Pros*: Fails fast with a clear error message; prevents silent disk fillup on the internal drive; deterministic.
  - *Cons*: Couples the pipeline to a specific external-volume topology; requires manual unmount checks for CI environments without the SSD.

**Option B**: **Environment-Driven Cache Path** — Read `SSD_HF_CACHE` from an env var (e.g., `PROXEMIC_HF_CACHE`) with a sane default and validate the parent directory exists before assignment.
  - *Pros*: Decouples code from hardware topology; CI and dev environments can override; aligns with twelve-factor configuration.
  - *Cons*: Adds an environment-config burden on contributors; existing scripts/runs depending on the hardcoded path must be updated.

**Option C**: **Promote Warning to Error** — Keep the hardcoded path but escalate the post-init check from `print("Warning: ...")` to `raise RuntimeError(...)` if the cache directory under `SSD_HF_CACHE/hub/...` does not exist after model download.
  - *Pros*: Minimal change; catches the failure mode after model download attempts.
  - *Cons*: Models may have already filled the internal drive by the time the error fires; reactive rather than preventive.

Your selection: Proceed with Option A.

---

### Issue 5: GPU/MPS Memory Not Released Across Long Batch Runs
**Status**: ⚠️ Confirmed Unresolved — Verified across `pipeline.py:44-77` (model init) and `pipeline.py:102-134` (`run` loop). Both `self.depth_estimator` (Depth Anything V2-Small, ~25M params) and `self.sam_estimator` (SAM ViT-Base, ~94M params) are instantiated once and held for the lifetime of `ProxemicKinematicsPipeline`. Neither `torch.mps.empty_cache()` (for the documented MPS backend on M4 Pro) nor `torch.cuda.empty_cache()` is called between videos or between frames. Across batches with 10-20 frame samples × multiple bystanders × dozens of videos, MPS unified-memory pressure can accumulate from cached intermediate activations and lead to allocation failures or thermal-induced throttling on the 24GB shared budget.

**Option A (recommended)**: **Per-Video `torch.mps.empty_cache()`** — Call `torch.mps.empty_cache()` (with a CUDA-equivalent guarded by `self.device == 'cuda'`) at the end of each `process_video` iteration in `run()`.
  - *Pros*: Bounds MPS cache growth at one-video granularity; minimal latency overhead (cache-flush is fast); negligible code change.
  - *Cons*: Does not address per-frame accumulation within a single long task; minor wall-clock cost per video.

**Option B**: **Per-Frame `torch.no_grad()` + Cache Flush** — Wrap depth and SAM inference in `torch.inference_mode()` (or `no_grad()`) and flush the MPS cache every N frames.
  - *Pros*: Tightest memory bound; eliminates gradient buffer overhead which `transformers.pipeline` may not disable by default.
  - *Cons*: More invasive; need to verify `pipeline()` honors the outer context; per-frame flush adds non-trivial overhead.

**Option C**: **Lazy Model Lifecycle** — Initialize models on first use within `process_video` and `del` them after each video, allowing Python GC + MPS to fully reclaim.
  - *Pros*: Strongest memory containment.
  - *Cons*: Pays full model-load cost (seconds for SAM, sub-second for Depth Anything) per video; severely degrades batch throughput.

Your selection: Proceed with Option A.

---

### Issue 6: Hardcoded Heuristic Constants Throughout the Vector Pipeline
**Status**: ⚠️ Confirmed Unresolved — Magic numbers governing the proxemic classification are inlined as literals: `noise_threshold = 15.0` at `pipeline.py:163`, `bbox_delta / 50.0` at `pipeline.py:424`, `-depth_delta * 2.0` at `pipeline.py:430`, weights `0.4`/`0.6` at `pipeline.py:431`, micro-movement deadband `0.05` at `pipeline.py:434`, and action thresholds `±0.3` at `pipeline.py:438-441`. None are exposed via constructor arguments, configuration, or class-level constants. Tuning the pipeline (e.g., adjusting sensitivity for older infants vs. adults, or recalibrating after switching to V1-Large per the documented fallback) requires editing source.

**Option A (recommended)**: **Class-Level Constants Block** — Hoist all heuristic constants to a `# --- Tuning Constants ---` block at the top of `ProxemicKinematicsPipeline` (e.g., `BBOX_NORM_PCT = 50.0`, `DEPTH_NORM_SCALE = 2.0`, `BBOX_WEIGHT = 0.4`, `DEPTH_WEIGHT = 0.6`, `MICROMOVEMENT_THRESHOLD = 0.05`, `APPROACH_THRESHOLD = 0.3`, `OPTICAL_FLOW_NOISE_THRESHOLD = 15.0`).
  - *Pros*: Centralizes tuning surface; easy to override via subclassing for ablations; zero runtime overhead; no schema change.
  - *Cons*: Still requires code edit to retune for production runs; not externally configurable per-batch.

**Option B**: **Config File (YAML/JSON)** — Load constants from a `proxemic_config.yaml` alongside the manifest path, with documented defaults.
  - *Pros*: Non-developers can retune; supports per-experiment configurations; auditable as artifacts.
  - *Cons*: Adds a config-loader dependency; tests must construct config fixtures; potential for misconfiguration drift.

**Option C**: **Constructor Arguments with Defaults** — Add named parameters (`bbox_norm_pct=50.0`, etc.) to `__init__` with sensible defaults.
  - *Pros*: Pythonic; tests can override per-test; type-hints document the tuning surface.
  - *Cons*: Constructor signature explosion (7+ new args); callers must thread through orchestration code.

Your selection: Proceed with Option A.