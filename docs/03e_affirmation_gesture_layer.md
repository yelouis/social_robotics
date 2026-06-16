# AI Task Breakdown: Affirmation Gesture Layer (03e)

## Objective
The **Affirmation Gesture Layer** parses the most explicit non-verbal heuristic an infant uses for moral validation: Head Nodding and Head Shaking. Even if an adult is smiling, a shaking head signals "No." This layer extracts rhythmic, high-frequency spatial oscillations from the bystander's head tracking data to explicitly classify affirmation or negation.

---

## 📥 Input Requirements
- **`03a_attention_result.json`** (required cross-layer): This layer acts as a direct mathematical extension of 03a. It prefers the **head-pose** vectors (`head_pitch_rad`/`head_yaw_rad`, from 03a's MediaPipe FaceLandmarker — Resolved #4) and falls back to the L2CS-Net **gaze** vectors (`pitch_rad`/`yaw_rad`) when head pose is unavailable, recording which via `signal_source`. The pipeline strictly requires this file and must fail-fast (`RuntimeError`) if missing or empty, to prevent silent failures and wasted compute.
- **`filtered_manifest.json`**: For reaction window boundaries.

---

## 🛠️ Implementation Strategy

### 1. Data Reuse Pipeline
Do not re-run inference on the videos. Load the raw 3D head pose arrays from Layer 03a. For each sample in the reaction window, we have `(timestamp, pitch, yaw)`.

### 2. Time-Series Signal Processing (SciPy)
Nodding and shaking have distinct frequency signatures—they are rhythmic oscillations occurring roughly between 1Hz and 3Hz.
- **Uniform Resampling**: Layer 03a utilizes an adaptive stride (e.g., 8 FPS baseline, up to 32 FPS bursts). Because `scipy.signal.filtfilt` requires uniform sampling intervals to avoid frequency distortion, we must interpolate the raw signals onto a fixed-dt grid using `scipy.interpolate.interp1d` before filtering.
- **NaN Gap Limiting (Model-Calibrated)**: When the gaze model loses face tracking, `NaN` gaps occur. While we linearly interpolate across these gaps, bridging long absences creates a straight-line ramp that, after detrending and bandpassing, can fabricate spurious oscillations. We track the `interpolated_fraction` and short-circuit bystanders exceeding a threshold. *Crucially, this threshold is re-keyed to the upstream model* (e.g., `0.3` for L2CS-Net, `0.2` for CrossGaze) to match each model's specific tracking-loss distribution profile.
- **Saccade Suppression (FPS-Gated Median Smoothing)**: We consume gaze vectors (`pitch_rad`, `yaw_rad`), which conflate rapid eye saccades with head rotation. To suppress saccadic impulses from aliasing into the 1-3Hz band, we apply a rank-order `medfilt(kernel_size=3)`. *Why it is gated:* This is strictly gated to activate only at high sampling rates (`fps >= 32.0`). At lower cadences, a 3-sample median window spans too much of a genuine nod's period and physically demolishes the tone.
- **Dynamic Bandpass Filtering**: We isolate human communication gestures (1-3Hz). However, because the effective FPS can be low, a static 1.5Hz cutoff might mathematically erase genuine 2.0Hz nods. We use a three-tier Nyquist-aware strategy that dynamically adjusts the frequency band based on the effective sampling rate.
- **Nodding & Shaking Detection**: Look for rhythmic variance. Use peak-finding (`find_peaks`) to identify rhythmic extrema on Pitch (nodding) and Yaw (shaking).
- **Ambiguous Gesture Classification**: If simultaneous pitch and yaw oscillations occur (a diagonal wobble), the gesture is classified as `ambiguous_wobble` when both confidence scores exceed `0.6` and are within a `0.15` delta. This prevents arbitrary bias toward nods or shakes.
- **Energy-Weighted Confidence**: A purely count-based confidence score treats vigorous nods and imperceptible micro-nods identically. The pipeline computes the RMS amplitude of the bandpass-filtered signal and applies it as a normalized multiplicative factor to the count-based confidence score.

### 3. Emotion Corollary
Combine this with Layer 03b. A "Smile" + "Nod" = Absolute positive validation. A "Smile" + "Shake" = Playful invalidation or disbelief. 

---

## 📤 Output Schema and Integration

To optimize resume cycles and make filtering decisions explicit, skipped videos (e.g., missing attention data or lacking tasks) are written as "sentinel" entries with a `skipped_reason` rather than being silently ignored.

**Example Output Data (`03e_affirmation_gesture_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03e_affirmation_gesture",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "per_person": [
        {
          "person_id": 0,
          "pitch_oscillation_hz": 2.1,
          "yaw_oscillation_hz": 0.2,
          "interpolated_fraction": 0.04,
          "gesture_detected": "affirming_nod",
          "confidence": 0.94
        }
      ]
    }
  ]
}
```
*(Sentinel Example: `{"video_id": "clip_2", "layer": "03e_affirmation_gesture", "tasks_analyzed": [], "skipped_reason": "no_attention_data"}`)*

## Verification & Validation Check
- **Singular Video Test**: Plot the Pitch and Yaw arrays explicitly on a matplotlib line graph for a known head-nod video. Visually identify the sine-wave signature of the nod on the Pitch axis and verify the SciPy peak-finding logic successfully counted the nods.
- **Batch Test**: Pass 50 clips through the signal processor. Assert that the script executes in milliseconds (as it uses pre-computed vectors) and gracefully handles `NaN` values where L2CS-Net lost tracking on the face, filling defaults safely to avoid breaking the Pandas merge step on the **Mac Studio (M4 Max, 64 GB unified memory)**.

## 🚀 Implementation Accomplishments (April 2026)

The Affirmation Gesture Layer has been implemented successfully in `src/layer_03e_affirmation_gesture/pipeline.py`.

- **Signal Extraction:** Built a pipeline that parses the `pitch_rad` and `yaw_rad` attention traces from `03a_attention_result.json` filtered strictly to the task's reaction window boundaries.
- **Bandpass Filtering:** Employed `scipy.signal.butter` and `filtfilt` to isolate the 1-3Hz frequencies characteristic of human nodding and shaking. A three-tier Nyquist-aware strategy selects the appropriate filter configuration based on the effective sampling rate.
- **Zero-Crossing Detection:** Used `find_peaks` to identify rhythmic extrema (peaks and troughs) with specific prominence thresholds to compute frequency oscillations safely without running inference on raw video.
- **Uniform Resampling:** Added `scipy.interpolate.interp1d` to resample non-uniform timestamps (from 03a's adaptive stride of 0.2s-0.5s) onto a fixed-dt grid before applying `filtfilt`, which requires uniform spacing.
- **Hard Dependency Validation:** The pipeline now raises a `RuntimeError` at startup if `03a_attention_result.json` is missing or empty, enforcing the documented hard dependency contract.
- **Ambiguous Gesture Classification:** Added `ambiguous_wobble` classification when both pitch and yaw oscillate with similar confidence (within 0.15 threshold), preventing arbitrary nod/shake bias.

## 🧪 Resolved Issues & Implementation Refinements

*(All historical implementation refinements regarding Nyquist limits, adaptive resampling, saccade suppression, interpolation thresholding, and missing-dependency enforcement have been standardized and integrated directly into the core Implementation Strategy.)*

1. **03e Read the Wearer-Climax Reaction Window but 03a Only Sampled Bystander-Detection Windows — 0/50 Yield (Resolved - June 14)**:
   - **Problem**: The June 14 50-clip smell test (`e2e_reports/2026_06_14_layer03e_50/`) scored **0/50**. `process_video` filtered 03a's `attention_trace` to the strict `task_reaction_window_sec` (wearer optical-flow climax ±1 s), but 03a's window-restricted sampling only runs gaze within ±2 s of each **bystander detection** timestamp, and detections sit a median ~6.8 s from the wearer's climax (docs/03d Resolved #1). So the reaction window held **zero** trace samples on 17 of the 23 clips with 03a data → `insufficient_trace`. This is the identical wearer-vs-bystander anchoring mismatch that 03b (Resolved #2) and 03d (Resolved #1) fixed; 03e never received it.
   - **Solution** (Option A): Added `_measurement_window()` mirroring 03d's `_bystander_measurement_window` (**June 14: both now delegate to the shared `src/shared/bystander_window.py` helper — see docs/03 Cross-Layer § Shared Helper**) — keep the reaction window when it already holds ≥ `MIN_TRACE_POINTS` trace samples; otherwise anchor to the bystander DETECTION nearest `task_climax_sec` ± `ANCHOR_SPAN_DETECTIONS`, padded by `WINDOW_PAD_SEC` (= 03a's 2 s sampling pad) and bounded by `MAX_ANCHOR_SPAN_SEC` (= 30 s). Each scored record now carries `measurement_window_sec` + `window_source`, and the sentinel reason is aggregated via `_aggregate_skip_reason` (`span_capped` / `insufficient_trace` / `mixed_skip`). **Validated on the June 14 re-run against the existing 03a traces (no 03a re-run needed): 0 → 20 scored clips, 83 person-task rows (70 `bystander_anchored`, 13 `reaction_window`), all anchored spans ≤ 28 s.** *(This restores yield; the now-visible gestures remain subject to Unresolved Issue 1 (gaze vs head pose) and the new NoFace-zero artifact (Unresolved Issue 2) — i.e. the fix makes 03e produce output, not yet trustworthy output.)*

2. **`filtfilt` padlen Crash on Short Windows Aborted the Whole Clip (Resolved - June 14)**:
   - **Problem**: The 6 June-14 clips whose window overlapped 03a's samples (the only scorable clips) all raised `ValueError: The length of the input vector x must be greater than padlen, which is 15` from `scipy.signal.filtfilt`: a windowed + resampled per-person signal of ~11–15 samples is shorter than the 2nd-order Butterworth bandpass `padlen` (15), and filtfilt **raises** rather than degrading. With no per-person `try/except`, the exception aborted the **entire clip** (no record, not even a sentinel) — one short-trace person poisoned every co-occurring person, so the run wrote 44 records not 50 and the scorable clips yielded nothing.
   - **Solution** (Option A): Added `_safe_filtfilt()` (returns `None` when `len(sig) <= 3·max(len(a),len(b))`); `_detect_oscillation` falls back to the raw-detrended peak-finding path on a too-short signal instead of raising. Wrapped the per-person body in `try/except` so any unexpected failure records that person's skip (`error`) and can never abort the clip. **Validated: the June 14 re-run produced 50 records with 0 crashes (empty error log).**

3. **NoFace / Lost-Tracking Samples Were Encoded as `pitch=yaw=0.0`, Fabricating Step-Edge "Gestures" (Resolved - June 14)**:
   - **Problem**: With Resolved #1/#2 in place the June 14 re-run emitted 27 gestures, but spot-checking showed they were artifacts, not head motion. 03a writes lost-tracking samples into `attention_trace` as `pitch_rad = yaw_rad = 0.0` with `target = "NoFace"` (**not `NaN`**), so `_fill_nan` / `interpolated_fraction` never flagged them (all 27 reported `interpolated_fraction = 0.0`) and the step transitions in and out of those zero-plateaus were counted by `find_peaks` as rhythmic extrema. The visually-verified exemplar `0780244d` pid5 (`affirming_nod`, conf 1.0) was a flat-zero plateau + a single spike + a 0→1.05-rad step — not an oscillation.
   - **Solution** (Option A, 03e-load-side variant — no 03a re-run): in `process_video`, samples whose `target == "NoFace"` are mapped to `NaN` when building the pitch/yaw arrays, so they flow through the existing `_fill_nan` + `interpolated_fraction` guard instead of forming real-valued steps. **Validated on the June 14 re-run against the existing 03a traces: detected gestures 27 → 6 (the NoFace step-edge false positives — including `0780244d` pid5 — are gone, that bystander now correctly `over_interpolated`); 7 clips became honest `over_interpolated` sentinels; `interpolated_fraction` now spans 0–0.27 (was uniformly 0.0).** The 6 survivors are gaze-derived and remain subject to Unresolved Issue 1. Covered by `test_noface_samples_treated_as_missing` and `test_small_noface_fraction_still_detects`.

4. **Nod/Shake Was Derived From GAZE, Not Head Pose — Confident False Nods From Eye Motion (Resolved - June 14)**:
   - **Problem**: 03e's premise (nod = pitch oscillation, shake = yaw oscillation) assumes HEAD orientation, but `process_video` read 03a's `pitch_rad`/`yaw_rad`, which are L2CS **gaze** angles (where the *eyes* point). Gaze moves with saccades / smooth-pursuit while the head is still, so non-nod eye motion aliased into the 1–3 Hz band — on the post-Resolved-#3 re-run all 6 surviving `affirming_nod`s were gaze artifacts, not head motion (e.g. `0780244d` pid5's head-crop strip showed the bystander looking around, not nodding).
   - **Solution** (Option A): New shared `src/shared/head_pose.py` (`HeadPoseEstimator`) runs MediaPipe **FaceLandmarker** on the same face crop 03a already uses for gaze, decomposing `facial_transformation_matrixes` → head Euler **pitch (nod) / yaw (shake)**. 03a emits additive `head_pitch_rad`/`head_yaw_rad` per trace sample (gated by `ENABLE_HEAD_POSE` + the `face_landmarker.task` asset; `null` when FaceLandmarker finds no face — its own tracking-loss). 03e now **prefers head pose** (`signal_source` provenance), mapping `null` head pose → `NaN` (Resolved #3 semantics) and falling back to gaze only when a window has no head pose at all. **Validated on the 13 gesture-producing clips (`e2e_reports/2026_06_14_layer03e_headpose/`): gaze false nods 6 → 1 — the 5 clips with any head-pose coverage are rejected (head pitch flat → `none`, or `over_interpolated`); the lone survivor (`0780244d` pid1) is an honest gaze-fallback, flagged `signal_source = "gaze"` because FaceLandmarker resolved no face in its window.** *(Amplitude re-tuning of the 1–3 Hz bands / `PEAK_PROMINENCE` / `RMS_THRESHOLD` for head-pose scale, and validation against true nod footage, are deferred — the Ego4D sample is thin on deliberate nods. The sparse head-pose coverage this surfaced is now Unresolved Issue 1.)*

### 🧪 Test Suite Results (25/25 Passed across `test_layer_03e.py` + `test_head_pose.py`)

A comprehensive verification suite in `tests/test_layer_03e.py` validates the following:
- **Nod/Shake/None Classification:** Synthetic sine-wave verification at ~2Hz.
- **NaN Resilience:** Gap patching using `np.interp` before filtration.
- **Hard Dependency Validation:** Raising `RuntimeError` on missing dependency files.
- **Ambiguous Wobble:** Correct classification of simultaneous equal-amplitude oscillations.
- **Non-Uniform Sampling:** Verification of detection accuracy after uniform resampling.
- **Upstream Gaze-Model Calibration:** `MAX_INTERPOLATED_FRACTION` re-keys to 0.3 / 0.2 / 0.25 against `processing_meta.model_used` across L2CS-Net, CrossGaze, 3DGazeNet, unknown-model, fallback-only, and missing-meta branches.
- **Read-Window Anchoring (Resolved #1):** keeps a dense reaction window, anchors (padded) to the climax-nearest detection when the window is empty, caps an over-long span, and recovers an offset trace end-to-end.
- **filtfilt Short-Signal Guard + Per-Person Isolation (Resolved #2):** `_safe_filtfilt` returns `None` below padlen; a short window falls back without crashing; a per-person exception yields a sentinel, not a clip abort.
- **NoFace-as-Missing (Resolved #3):** `target='NoFace'` samples become NaN — a NoFace-dominated window is rejected (`over_interpolated`), while a few NoFace samples are bridged without killing a genuine nod.
- **Head-Pose Preference (Resolved #4):** 03e prefers `head_*` over gaze (`signal_source` provenance) — an oscillating gaze with a flat head scores `none`; a real head-pitch oscillation is a nod; `null` head pose is missing (NaN), not zero; old traces without `head_*` fall back to gaze.
- **Head-Pose Estimator (`test_head_pose.py`):** Euler decomposition (pitch = X-axis nod, yaw = Y-axis shake); degrades to `None` when the asset is missing (never raises).

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Head-Pose Coverage Is Sparse on Small Bystander Faces (~8% of samples)
**Status**: ⚠️ Confirmed Unresolved — Surfaced by the June 14 head-pose run (`e2e_reports/2026_06_14_layer03e_headpose/`, Resolved #4): of **10,486** trace samples across 12 clips, MediaPipe FaceLandmarker resolved head pose on only **8%** — 42% were already `NoFace` (no face at all for the gaze gate) and **50% were face-present-but-no-landmarks** (BlazeFace's looser gate passed, but FaceLandmarker found no mesh on the small / steeply-angled / motion-blurred bystander face). Consequently most per-person windows have too little head pose to score: on the gesture clips only **2 of 9** scored windows used head pose; the rest fell back to gaze (`signal_source="gaze"`, where the 1 residual gaze false-nod lives) or became `over_interpolated`. So Resolved #4 correctly rejects gaze artifacts *where head pose exists*, but head pose rarely exists on this egocentric corpus — the layer is now honest but low-yield. *(Distinct from the deferred amplitude re-tuning noted in Resolved #4: this is about head-pose **availability**, not threshold scale.)*

**Option A (recommended)**: **Upscale the face crop before FaceLandmarker.** FaceLandmarker needs ~face ≥ ~64 px of mesh detail; bystander crops are often smaller. Resize the crop (e.g. to a fixed 256–512 px short edge, or only when below a size floor) before `detect`, trading compute for landmark recall on small faces.
  - *Pros*: Cheap, inside the existing pass; directly targets the 50% face-present-no-landmark bucket; no new model/asset.
  - *Cons*: Upscaled tiny faces yield noisier pose; a recall/precision knob to validate; won't help genuinely unresolvable faces.

**Option B**: **Swap FaceLandmarker for a small-face-robust 6DoF head-pose model** (e.g. 6DRepNet / WHENet) that regresses pose directly from a coarse crop without a dense landmark mesh.
  - *Pros*: Designed for direct head-pose regression; typically more robust on small / low-detail faces; no mesh-detection gate.
  - *Cons*: New model asset + dependency; its own amplitude scale to re-tune; heavier than the already-present MediaPipe Tasks.

**Option C**: **Accept gaze-fallback-with-provenance as the honest floor.** Keep Resolved #4 as-is: head pose where available, gaze (flagged `signal_source="gaze"`) elsewhere; let downstream weight gaze-derived gestures lower.
  - *Pros*: Zero new work; fully honest (provenance already emitted); no false confidence.
  - *Cons*: Most gestures stay gaze-derived (the very signal Resolved #4 set out to replace); the low head-pose yield persists.

Your selection: _____

> **June 14 update:** the smell-test-blocking findings (window mismatch; `filtfilt` crash) **and** the NoFace-zero step-edge artifact are now **resolved above** (Resolved #1–#3). With them fixed, the June 14 re-run yields **6 detected gestures** (down from 27) — all now gaze-derived, i.e. squarely this Issue 1. A dedicated head-pose source (Option A) is what remains to make 03e's output *trustworthy*, not merely present.
