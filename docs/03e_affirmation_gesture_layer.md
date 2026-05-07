# AI Task Breakdown: Affirmation Gesture Layer (03e)

## Objective
The **Affirmation Gesture Layer** parses the most explicit non-verbal heuristic an infant uses for moral validation: Head Nodding and Head Shaking. Even if an adult is smiling, a shaking head signals "No." This layer extracts rhythmic, high-frequency spatial oscillations from the bystander's head tracking data to explicitly classify affirmation or negation.

---

## 📥 Input Requirements
- **`03a_attention_result.json`** (required cross-layer): This layer acts as a direct mathematical extension of 03a. We reuse the 3D Pitch/Yaw vectors extracted by L2CS-Net during the attention phase.
- **`filtered_manifest.json`**: For reaction window boundaries.

---

## 🛠️ Implementation Strategy

### 1. Data Reuse Pipeline
Do not re-run inference on the videos. Load the raw 3D head pose arrays from Layer 03a. For each sample in the reaction window, we have `(timestamp, pitch, yaw)`.

### 2. Time-Series Signal Processing (SciPy)
Nodding and shaking have distinct frequency signatures—they are rhythmic oscillations occurring roughly between 1Hz and 3Hz.
- Use `scipy.signal` to apply a Bandpass Filter to isolate frequencies typical of human communication gestures.
- **Nodding Detection**: Look for rhythmic variance (peaks and valleys) exclusively on the **Pitch** (up/down) axis. Use peak-finding algorithms to count zero-crossings. If > 2 oscillating zero-crossings occur vertically within 1.5 seconds, flag as `nodding`.
- **Shaking Detection**: Apply the exact same logic to the **Yaw** (left/right) axis. If rhythmic oscillation breaches the variance threshold laterally, flag as `shaking`.

### 3. Emotion Corollary
Combine this with Layer 03b. A "Smile" + "Nod" = Absolute positive validation. A "Smile" + "Shake" = Playful invalidation or disbelief. 

---

## 📤 Output Schema and Integration
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
          "pitch_variance_hz": 2.1,
          "yaw_variance_hz": 0.2,
          "gesture_detected": "affirming_nod",
          "confidence": 0.94
        }
      ]
    }
  ]
}
```

## Verification & Validation Check
- **Singular Video Test**: Plot the Pitch and Yaw arrays explicitly on a matplotlib line graph for a known head-nod video. Visually identify the sine-wave signature of the nod on the Pitch axis and verify the SciPy peak-finding logic successfully counted the nods.
- **Batch Test**: Pass 50 clips through the signal processor. Assert that the script executes in milliseconds (as it uses pre-computed vectors) and gracefully handles `NaN` values where L2CS-Net lost tracking on the face, filling defaults safely to avoid breaking the Pandas merge step on the **Mac mini M4 Pro**.

## 🚀 Implementation Accomplishments (April 2026)

The Affirmation Gesture Layer has been implemented successfully in `src/layer_03e_affirmation_gesture/pipeline.py`.

- **Signal Extraction:** Built a pipeline that parses the `pitch_rad` and `yaw_rad` attention traces from `03a_attention_result.json` filtered strictly to the task's reaction window boundaries.
- **Bandpass Filtering:** Employed `scipy.signal.butter` and `filtfilt` to isolate the 1-3Hz frequencies characteristic of human nodding and shaking. A three-tier Nyquist-aware strategy selects the appropriate filter configuration based on the effective sampling rate.
- **Zero-Crossing Detection:** Used `find_peaks` to identify rhythmic extrema (peaks and troughs) with specific prominence thresholds to compute frequency oscillations safely without running inference on raw video.
- **Uniform Resampling:** Added `scipy.interpolate.interp1d` to resample non-uniform timestamps (from 03a's adaptive stride of 0.2s-0.5s) onto a fixed-dt grid before applying `filtfilt`, which requires uniform spacing.
- **Hard Dependency Validation:** The pipeline now raises a `RuntimeError` at startup if `03a_attention_result.json` is missing or empty, enforcing the documented hard dependency contract.
- **Ambiguous Gesture Classification:** Added `ambiguous_wobble` classification when both pitch and yaw oscillate with similar confidence (within 0.15 threshold), preventing arbitrary nod/shake bias.

## 🧪 Resolved Issues & Implementation Refinements

1. **Bandpass Cutoff at Low FPS (Resolved - April 29)**:
   - **Problem**: Layer 03a uses an adaptive stride that can drop to ~2.5-5 FPS, resulting in a low Nyquist frequency. A static 1.5Hz bandpass filter cutoff would mathematically erase the 2.0Hz signature of a standard head nod, leading to false negatives.
   - **Solution**: Implemented a three-tier Nyquist-aware bandpass strategy that dynamically adjusts the frequency band based on the effective sampling rate, ensuring signals are preserved even at lower FPS.

2. **Non-Uniform Sampling Breaks `filtfilt` (Resolved - April 29)**:
   - **Problem**: `scipy.signal.filtfilt` assumes uniform sampling intervals. However, the adaptive stride produces intervals varying from 0.2s to 0.5s, which distorts the frequency response when using a mean sampling rate.
   - **Solution**: Added a uniform resampling step using `scipy.interpolate.interp1d` before filtering. The pipeline now interpolates raw signals onto a fixed-dt grid, ensuring the filter operates on correctly-spaced samples.

3. **Silent Failure on Missing Upstream Dependencies (Resolved - April 29)**:
   - **Problem**: The layer has a hard dependency on Layer 03a's `attention_trace`. Previously, it silently skipped videos with "No attention data found" if the dependency was missing, violating the architectural contract and wasting compute time.
   - **Solution**: Updated the `__init__` method to raise a `RuntimeError` immediately if the required `03a_attention_result.json` is missing or unparseable, enforcing a "fail-fast" behavior.

4. **Ambiguous Gesture Classification (Resolved - April 29)**:
   - **Problem**: Simultaneous pitch and yaw oscillations (e.g., a diagonal head wobble) were arbitrarily classified as either a nod or a shake based on code branch order, creating biased results.
   - **Solution**: Introduced a tie-breaking logic that classifies the gesture as `ambiguous_wobble` when both confidence scores exceed 0.6 and are within a 0.15 threshold of each other.

5. **Nyquist Ceiling on Fast Nods (Resolved - May 05)**:
   - **Problem**: Layer 03a's adaptive stride sampled at ~2.5-5 FPS. By the Nyquist-Shannon theorem, the maximum detectable frequency was 1.25 Hz. Standard communicative nods occur at 1.5-3 Hz, meaning a significant portion of genuine nods were physically undetectable.
   - **Solution**: Changed 03a's default sampling stride to a fixed 8 FPS (0.125s interval) globally, raising the Nyquist ceiling to 4 Hz and ensuring fast communicative nods are accurately sampled without relying solely on downstream layer interpolation.

6. **Confidence Formula Energy Weighting (Resolved - May 05)**:
   - **Problem**: The `confidence` metric was based solely on the count of zero-crossings detected by `find_peaks`, treating vigorous nods and barely-perceptible micro-nods identically, hindering downstream emotion correlation.
   - **Solution**: Implemented an RMS Amplitude Weighting approach in Layer 03e. The pipeline now computes the RMS amplitude of the bandpass-filtered signal and applies a normalized multiplicative factor to the count-based confidence score, effectively distinguishing emphatic agreement from ambiguous movement.

7. **Misleading Field Semantics (`*_variance_hz`) (Resolved - May 05)**:
   - **Problem**: The output schema fields `pitch_variance_hz` and `yaw_variance_hz` contained oscillation frequencies rather than statistical variance measures, causing semantic confusion.
   - **Solution**: Addressed the technical debt by introducing `pitch_oscillation_hz` and `yaw_oscillation_hz` alongside the legacy fields in the output schema of Layer 03e, following an additive-only schema evolution pattern to prevent downstream breakage while rectifying the nomenclature.

8. **Gaze Vectors vs. Head Pose Vectors Conflation (Resolved - May 05)**:
   - **Problem**: The layer consumed `pitch_rad` and `yaw_rad` from `03a_attention_result.json`, which represent L2CS-Net gaze direction vectors rather than head orientation. Fast eye saccades independent of head rotation could produce pitch/yaw oscillations mimicking nodding/shaking, leading to false positives.
   - **Solution**: Implemented Gaze-to-Head Proxy Filtering. Added a low-pass pre-filter in `src/layer_03e_affirmation_gesture/pipeline.py` before the bandpass to suppress fast saccadic oscillations. This reliably attenuates rapid eye movements while preserving the amplitude of genuine, slower head nods, without requiring additional heavy model dependencies.

9. **Upstream 03a Reverted to Adaptive Sampling (Resolved - May 07)**:
   - **Problem**: This layer's prior Resolved entries (#1 "Bandpass Cutoff at Low FPS", #2 "Non-Uniform Sampling Breaks `filtfilt`", #5 "Nyquist Ceiling on Fast Nods") all assumed a specific upstream cadence — first an adaptive 0.2 s–0.5 s stride, then a fixed 8 FPS stride after the 03a stabilization. Layer 03a has now been reverted to an adaptive `0.125 s` baseline (8 FPS) with 16 FPS bursts on attention-score deltas > 0.3 (see 03a Resolved Issue #12). If 03e's Nyquist tier and `interp1d` resampling logic had been tuned to the *fixed* 8 FPS assumption, the new non-uniform stride could mis-trigger or skip resampling.
   - **Solution**: No code changes were required. The 8 FPS baseline preserved by 03a is the *floor* of the new adaptive scheme, so 03e's Nyquist-aware bandpass tier (which already accommodated the original 0.2 s–0.5 s adaptive trace) and the `scipy.interpolate.interp1d` resampling step (already engaged unconditionally) both continue to behave correctly. Bursts to 16 FPS only ever *add* samples; they cannot drop the effective rate below the 4 Hz Nyquist ceiling on which Resolved Issue #5's analysis rests. This entry exists purely to record the upstream contract change so that future audits do not assume a uniform stride.

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Gaze-to-Head Low-Pass Pre-Filter is Functionally Redundant
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:235-260`. The "Gaze-to-Head Proxy Filtering" (Resolved Issue 8) installs a 2nd-order Butterworth low-pass at cutoff `3.0 / nyq`, but only when `nyq > 3.0`. In that exact branch, the subsequent bandpass at `[1.0/nyq, 3.0/nyq]` already attenuates everything above 3 Hz. The low-pass therefore adds zero suppression of saccadic oscillations — its passband (0–3 Hz) is a strict superset of the bandpass passband (1–3 Hz). In the marginal-Nyquist tier (`1.5 < nyq <= 3.0`) and the no-filter tier (`nyq <= 1.5`), the low-pass does not fire at all. The documented motivation — "suppress fast saccadic oscillations [that] could produce pitch/yaw oscillations mimicking nodding/shaking" — is not actually addressed: any saccadic energy that survived sampling and aliased into the 1–3 Hz band is preserved by *both* filters since it now lives within the passband. The fix is structurally inert.

**Option A (recommended)**: **Replace the Pre-Filter With a Median-Smoothing Step on the Raw Trace** — Apply `scipy.signal.medfilt` (kernel=3 or 5) to the raw `pitch_rad`/`yaw_rad` traces *before* detrending. Saccades manifest as single-sample spikes superimposed on the slower head-pose drift; a small-kernel median filter removes them while preserving the smoother nod/shake oscillation envelope.
  - *Pros*: Targets the actual failure mode (impulsive saccades); cheap to compute; no Nyquist constraint; preserves the bandpass downstream.
  - *Cons*: May slightly attenuate sharp nod transitions; requires choosing kernel size; not a true frequency-domain operation.

**Option B**: **Use a True Anti-Saccade Notch** — Add a notch filter centered around typical saccadic frequencies (~5–10 Hz) that fires *before* the bandpass. Requires `nyq > 10 Hz` to be meaningful, which the current 8 FPS stride does not provide (Nyquist=4 Hz).
  - *Pros*: Frequency-domain principled; explicit saccade attenuation.
  - *Cons*: Requires 03a to bump sampling above 20 FPS (re-opens Issue 5 trade-off); extra tier in the existing three-tier filter cascade.

**Option C**: **Adopt OpenFace/MediaPipe Head-Pose Vectors Instead of L2CS Gaze Vectors** — Replace the gaze-derived `pitch_rad`/`yaw_rad` from 03a with explicit head-pose estimates from a head-pose-only model, eliminating the gaze-vs-head conflation at its source.
  - *Pros*: Architecturally correct; no filter tricks needed; aligns with the original algorithmic intent ("3D head pose arrays").
  - *Cons*: Heavy refactor of 03a; new model dependency; loses the "Data Reuse Pipeline" benefit that motivates this layer.

Your selection: Proceed with Option A.

---

### Issue 2: Skipped Videos Reprocessed on Every Resume
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:84-102`. When `process_video` returns `None` (no `att_entry` for the video at line 110, no `identified_tasks` at line 115, all `per_person_results` filtered out at line 208, or every reaction window has < 5 trace points at line 139), the `if result:` guard skips both `results.append(...)` and `self.processed_ids.add(video_id)`. Subsequent resume runs reload the same `registry`, fail the `if video_id in self.processed_ids` check, and re-execute the full attention-trace lookup, NaN-filling, interp1d resampling, and butterworth filtering — only to discard the result again. For batches that share the upstream 03a result with thousands of videos but only a fraction with valid bystander tasks, this wastes substantial CPU per resume.

**Option A (recommended)**: **Sentinel-Record Tracking** — When `process_video` returns `None`, write a sentinel record (`{"video_id": ..., "layer": "03e_affirmation_gesture", "tasks_analyzed": [], "skipped_reason": "no_attention_data" | "no_tasks" | "insufficient_trace"}`) to `results` and add to `processed_ids`.
  - *Pros*: Persists skip decisions; downstream consumers gain visibility into *why* a video was excluded; resume cost drops to O(JSON-load + set-membership).
  - *Cons*: Output JSON inflates with empty entries; downstream consumers must filter on `tasks_analyzed` length or `skipped_reason`.

**Option B**: **Skip Manifest Sidecar** — Maintain `03e_skipped.json` listing video_ids that produced no output, short-circuit them at the top of the per-entry loop alongside the `processed_ids` check.
  - *Pros*: Keeps the main result JSON clean; explicit skip log is easy to audit.
  - *Cons*: Two-file state machine to maintain; risk of skip-manifest/result-manifest drift if writes aren't atomic.

**Option C**: **Always-Mark-Processed Policy** — Always add `video_id` to `processed_ids` after `process_video` returns, and persist `processed_ids` to disk as a separate JSON.
  - *Pros*: Cleanly decouples "did we attempt this?" from "did it produce output?"; minimal JSON inflation.
  - *Cons*: Adds a third state file; tests must mock or write the new file; loses the rationale for *why* a video was skipped.

Your selection: Proceed with Option A.

---

### Issue 3: Linear NaN Interpolation Has No Gap-Length Limit
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:217-222`. `_fill_nan` uses `np.interp` to linearly bridge any contiguous NaN run, regardless of duration. When L2CS-Net (03a's gaze model) loses face tracking for an extended interval — common during partial occlusions, motion blur, or extreme head rotations — the resulting trace contains long NaN runs (e.g., 1–2 seconds, equivalent to 8–16 samples at 8 FPS). Linear interpolation across these gaps produces a straight line that (a) silently smooths over genuine head motion that was happening during the gap and (b) introduces an artificial low-frequency ramp that, after detrending and filtering, can either suppress or fabricate oscillations. The pipeline has no telemetry on how much of each window was interpolated.

**Option A (recommended)**: **Reject Windows With Excessive Interpolation** — Compute `interpolated_fraction = mask.sum() / len(arr)` and short-circuit (`return None` for the bystander) when this exceeds a threshold (e.g., 0.3). Surface the metric in the output schema as `interpolated_fraction` for downstream filtering.
  - *Pros*: Prevents fabricated detections from sparse traces; explicit data-quality flag aligns with the documented "graceful NaN handling" rationale; cheap.
  - *Cons*: Drops borderline-valid windows; requires picking a threshold; bystanders with intermittent tracking may yield zero results.

**Option B**: **Consecutive-Gap Cap** — Detect runs of NaN longer than N samples (e.g., 4 samples = 0.5s at 8 FPS) and either (i) reject the window or (ii) split the trace into pre-gap / post-gap segments and analyze each independently.
  - *Pros*: Catches localized tracking outages; preserves data on either side of the gap.
  - *Cons*: Complicates window slicing; oscillation-frequency estimates degrade on short segments.

**Option C**: **Cubic-Spline Interpolation Instead of Linear** — Use `scipy.interpolate.CubicSpline` to bridge NaN runs with a smoother curve more representative of natural head motion.
  - *Pros*: Less artifactual ramping; smoother frequency response.
  - *Cons*: Risks oscillating splines for long gaps; does not address the root issue (no signal in the gap); only papers over the symptom.

Your selection: Proceed with Option A.

---

### Issue 4: Hardcoded Heuristic Constants Across the Detection Path
**Status**: ⚠️ Confirmed Unresolved — Magic numbers governing classification decisions are inlined as literals: minimum trace length `5` at `pipeline.py:139`, peak prominence `0.03` at `pipeline.py:270-271`, std-dev floor `0.02` at `pipeline.py:266`, frequency band `0.5 <= est_hz <= 4.0` at `pipeline.py:281`, count-confidence formula `0.5 + (total_extrema * 0.1)` at `pipeline.py:282`, RMS threshold `0.05` at `pipeline.py:284`, gesture decision threshold `0.6` at `pipeline.py:182, 185, 188`, and ambiguity threshold `0.15` at `pipeline.py:182`. None are exposed via constructor arguments, configuration, or class-level constants. Any retuning — e.g., adjusting sensitivity for high-noise environments, recalibrating after a 03a model swap, or comparing rad-amplitude conventions across datasets — requires editing source.

**Option A (recommended)**: **Class-Level Tuning Constants Block** — Hoist all heuristic constants into a `# --- Detection Tuning ---` block at the top of `AffirmationGesturePipeline` (e.g., `MIN_TRACE_POINTS = 5`, `PEAK_PROMINENCE = 0.03`, `STD_DEV_FLOOR = 0.02`, `FREQ_BAND = (0.5, 4.0)`, `RMS_THRESHOLD = 0.05`, `GESTURE_DECISION_THRESHOLD = 0.6`, `AMBIGUITY_DELTA = 0.15`).
  - *Pros*: Centralizes tuning surface; easy to override via subclassing for ablations; zero runtime overhead; no schema change.
  - *Cons*: Still requires code edit to retune in production; not externally configurable per-batch.

**Option B**: **Config File (YAML/JSON)** — Load constants from a `affirmation_gesture_config.yaml` co-located with the manifest path, with documented defaults.
  - *Pros*: Non-developers can retune; supports per-experiment configurations; auditable as artifacts.
  - *Cons*: Adds a config-loader dependency; tests must construct config fixtures; potential for misconfiguration drift.

**Option C**: **Constructor Arguments with Defaults** — Add named parameters to `__init__` with sensible defaults pulled from the current literal values.
  - *Pros*: Pythonic; tests can override per-test; type-hints document the tuning surface.
  - *Cons*: Constructor-signature explosion (7+ new args); orchestration code must thread parameters through.

Your selection: Proceed with Option A.

---

### Issue 5: Legacy `*_variance_hz` Fields Duplicate `*_oscillation_hz` With No Deprecation Path
**Status**: ⚠️ Confirmed Unresolved — Verified in `pipeline.py:194-197`. Resolved Issue 7 introduced `pitch_oscillation_hz` and `yaw_oscillation_hz` alongside the misnamed `pitch_variance_hz` and `yaw_variance_hz`, both populated with identical values via `round(pitch_var, 2)` / `round(yaw_var, 2)`. The "additive-only schema evolution" prevented downstream breakage, but no deprecation timeline, schema-version field, or downstream-consumer migration plan was recorded. The duplicate fields will persist indefinitely in every output JSON, doubling the per-person field count for these metrics and inviting confusion in future audits about which field is authoritative — especially if a future change recalculates only one of them.

**Option A (recommended)**: **Schema Version + Deprecation Window** — Add a top-level `"schema_version": "1.1"` to each result entry, document that `*_variance_hz` is deprecated as of 1.1 and will be removed in 2.0, and grep the codebase for downstream consumers of the legacy fields. Once consumers migrate, delete the legacy fields.
  - *Pros*: Explicit migration contract; auditable in version control; aligns with semantic-versioning practice.
  - *Cons*: Requires coordinating with downstream layers (03c, 03f, ensemble); schema-version plumbing must be honored elsewhere.

**Option B**: **Immediate Removal** — Delete `pitch_variance_hz` / `yaw_variance_hz` now and accept downstream breakage where it occurs.
  - *Pros*: Simplest resolution; eliminates duplicate state; smaller output JSON.
  - *Cons*: Breaks any consumer still on the legacy field name; reverts the rationale for additive-only evolution; risks silent KeyError in production runs.

**Option C**: **Computed Property at Read Time** — Stop writing `*_variance_hz` to disk; instead, expose a thin `result_loader.py` helper that synthesizes the legacy field on read for backward compatibility.
  - *Pros*: Output JSON is clean; legacy consumers still work via the loader.
  - *Cons*: Requires consumers to adopt the loader; couples write-path and read-path; adds a maintenance burden.

Your selection: Proceed with Option B.

### 🧪 Test Suite Results (6/6 Passed)

A comprehensive verification suite in `tests/test_layer_03e.py` validates the following:
- **Nod/Shake/None Classification:** Synthetic sine-wave verification at ~2Hz.
- **NaN Resilience:** Gap patching using `np.interp` before filtration.
- **Hard Dependency Validation:** Raising `RuntimeError` on missing dependency files.
- **Ambiguous Wobble:** Correct classification of simultaneous equal-amplitude oscillations.
- **Non-Uniform Sampling:** Verification of detection accuracy after uniform resampling.
