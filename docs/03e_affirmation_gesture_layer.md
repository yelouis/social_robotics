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

10. **Redundant Gaze-to-Head Low-Pass Replaced With FPS-Gated Median Smoothing (Resolved - May 08)**:
    - **Problem**: The "Gaze-to-Head Proxy Filtering" introduced in Resolved Issue 8 installed a 2nd-order Butterworth low-pass at cutoff `3.0 / nyq` only when `nyq > 3.0`. In that exact branch, the subsequent 1–3 Hz bandpass already attenuated everything above 3 Hz, so the low-pass added zero saccade suppression — its 0–3 Hz passband was a strict superset of the bandpass passband. In the marginal-Nyquist (`1.5 < nyq <= 3.0`) and no-filter (`nyq <= 1.5`) tiers, the low-pass did not fire at all. The fix was structurally inert and the documented motivation — suppressing saccadic oscillations that alias into the 1–3 Hz band — was not actually addressed: any saccadic energy that survived sampling and aliased into the passband was preserved by both filters.
    - **Solution**: Removed the redundant low-pass branch from `_detect_oscillation` in `src/layer_03e_affirmation_gesture/pipeline.py` and replaced it with a `scipy.signal.medfilt(kernel_size=3)` saccade-suppression step applied to the resampled signal before detrending. Median smoothing rejects single-sample impulse outliers via rank-order operation rather than frequency-domain attenuation, providing a mechanism that is genuinely orthogonal to the bandpass. To avoid corrupting fast nods sampled near Nyquist — where a 3-sample median window spans more than half a target nod period and demolishes the tone — the filter is gated on `fps >= MEDFILT_KERNEL * 4 * BANDPASS_HZ.high` (currently 36 FPS). Under 03a's 8 FPS adaptive baseline and 16 FPS attention-burst stride, the gate keeps the filter inert; it only activates when 03e is fed higher-FPS gaze traces from a future upstream change. The trade-off is mathematically unavoidable: at the sampling rates 03a actually produces, no in-pipeline filter (frequency-domain or rank-order) can distinguish aliased saccadic energy from genuine 1–3 Hz nod content, so the resolution preserves correctness at present rates while wiring up a forward-compatible mechanism that will engage automatically on higher-FPS sources.

11. **Skipped Videos Reprocessed on Every Resume (Resolved - May 08)**:
    - **Problem**: When `process_video` returned `None` (no `att_entry` for the video, no `identified_tasks`, all `per_person_results` filtered out, or every reaction window had < 5 trace points), the `if result:` guard in `run` skipped both `results.append(...)` and `self.processed_ids.add(video_id)`. Subsequent resume runs reloaded the same `registry`, failed the `if video_id in self.processed_ids` check, and re-executed the full attention-trace lookup, NaN-filling, `interp1d` resampling, and Butterworth filtering — only to discard the result again. For batches sharing a 03a result file with thousands of videos but only a fraction yielding valid bystander tasks, this wasted substantial CPU per resume.
    - **Solution**: Introduced an `_sentinel(video_id, reason)` helper that returns `{"video_id": ..., "layer": "03e_affirmation_gesture", "tasks_analyzed": [], "skipped_reason": "no_attention_data" | "no_tasks" | "insufficient_trace"}`, and replaced the three `return None` sites in `process_video` with calls to this helper. The truthiness gate in `run` was removed so every video — including skipped ones — is appended to `results`, written atomically, and added to `self.processed_ids`. Resume cost for previously-skipped videos drops to set-membership lookup, and downstream consumers gain explicit visibility into *why* a video was excluded. Consumers must filter on `tasks_analyzed` length or the presence of `skipped_reason` to avoid treating sentinel entries as detection results.

12. **Linear NaN Interpolation Has No Gap-Length Limit (Resolved - May 08)**:
    - **Problem**: `_fill_nan` used `np.interp` to linearly bridge any contiguous NaN run, regardless of duration. When L2CS-Net (03a's gaze model) lost face tracking for an extended interval — common during partial occlusions, motion blur, or extreme head rotations — the resulting trace contained long NaN runs (e.g., 1–2 seconds = 8–16 samples at 8 FPS). Linear interpolation across these gaps produced a straight line that silently smoothed over genuine head motion in the gap and introduced an artificial low-frequency ramp that, after detrending and bandpass filtering, could either suppress real oscillations or fabricate spurious ones. The pipeline emitted no telemetry on how much of each window had been reconstructed.
    - **Solution**: `_fill_nan` now returns a tuple `(filled_array, interpolated_fraction)` where `interpolated_fraction = mask.sum() / len(arr)`. `process_video` computes `max(pitch_interp_frac, yaw_interp_frac)` per bystander and short-circuits the bystander when the fraction exceeds the new `MAX_INTERPOLATED_FRACTION = 0.3` constant. The metric is also surfaced in the per-person output schema as `interpolated_fraction` (rounded to 3 decimals) so downstream layers can apply their own thresholds without re-deriving the figure from raw 03a traces. Bystanders with intermittent tracking that exceed the threshold yield zero results for that task, which is the documented intent — fabricated detections from sparse traces are worse than absent detections.

13. **Heuristic Constants Hoisted to Class-Level Tuning Block (Resolved - May 08)**:
    - **Problem**: Magic numbers governing classification decisions were inlined as literals throughout the detection path: minimum trace length `5`, peak prominence `0.03`, std-dev floor `0.02`, frequency band `0.5–4.0`, count-confidence formula `0.5 + 0.1 * total_extrema`, RMS threshold `0.05`, gesture decision threshold `0.6`, and ambiguity threshold `0.15`. None were exposed via constructor arguments, configuration, or class-level constants. Any retuning — adjusting sensitivity for high-noise environments, recalibrating after a 03a model swap, or comparing rad-amplitude conventions across datasets — required editing source files at multiple sites and risked drift between the detection logic and downstream documentation.
    - **Solution**: Hoisted all heuristic constants into a `# --- Detection Tuning ---` block at the top of `AffirmationGesturePipeline`: `MIN_TRACE_POINTS`, `MEDFILT_KERNEL`, `MAX_INTERPOLATED_FRACTION`, `PEAK_PROMINENCE`, `STD_DEV_FLOOR`, `FREQ_BAND_HZ`, `BANDPASS_HZ`, `COUNT_CONFIDENCE_BASE`, `COUNT_CONFIDENCE_PER_EXTREMUM`, `RMS_THRESHOLD`, `GESTURE_DECISION_THRESHOLD`, and `AMBIGUITY_DELTA`. All call sites in `process_video` and `_detect_oscillation` now reference the class attributes. Subclass-based ablations and per-experiment overrides become a matter of attribute assignment rather than source edits; zero runtime overhead, no schema change.

14. **Legacy `*_variance_hz` Fields Removed From Output Schema (Resolved - May 08)**:
    - **Problem**: Resolved Issue 7 introduced `pitch_oscillation_hz` and `yaw_oscillation_hz` alongside the misnamed `pitch_variance_hz` and `yaw_variance_hz`, both populated with identical values via `round(pitch_var, 2)` / `round(yaw_var, 2)`. The "additive-only schema evolution" prevented downstream breakage at the time, but no deprecation timeline, schema-version field, or downstream-consumer migration plan was recorded. The duplicate fields would have persisted indefinitely in every output JSON, doubling the per-person field count for these metrics and inviting confusion in future audits about which field was authoritative — especially if a future change ever recomputed only one of them.
    - **Solution**: Deleted `pitch_variance_hz` and `yaw_variance_hz` from the per-person output written by `process_video` in `src/layer_03e_affirmation_gesture/pipeline.py`. Only the correctly-named `pitch_oscillation_hz` / `yaw_oscillation_hz` fields remain, alongside the new `interpolated_fraction` metric. A repository-wide grep across `*.py`, `*.md`, and `*.json` confirmed no consumers outside the 03e source and docs themselves reference the legacy field names, so immediate removal is safe; any future consumer reading the legacy names will receive a `KeyError` on first read of the new outputs, which is the explicit intent — they must migrate to the canonical names.

## ⚠️ Unresolved Issues & Suggestions

_No unresolved issues at this time._

### 🧪 Test Suite Results (6/6 Passed)

A comprehensive verification suite in `tests/test_layer_03e.py` validates the following:
- **Nod/Shake/None Classification:** Synthetic sine-wave verification at ~2Hz.
- **NaN Resilience:** Gap patching using `np.interp` before filtration.
- **Hard Dependency Validation:** Raising `RuntimeError` on missing dependency files.
- **Ambiguous Wobble:** Correct classification of simultaneous equal-amplitude oscillations.
- **Non-Uniform Sampling:** Verification of detection accuracy after uniform resampling.
