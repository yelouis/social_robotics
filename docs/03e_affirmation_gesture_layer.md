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

## ⚠️ Unresolved Issues & Suggestions

### Issue 1: Gaze Vectors vs. Head Pose Vectors Conflation
**Status**: ⚠️ Confirmed Unresolved — Verified in the pipeline code: the layer consumes `pitch_rad` and `yaw_rad` from `03a_attention_result.json`, which are L2CS-Net **gaze direction** vectors (where the person is looking), not **head orientation** angles (where the head is physically pointed). Eye saccades — rapid eye movements independent of head rotation — can produce pitch/yaw oscillations that mimic head nodding/shaking patterns. For example, a person reading text left-to-right would produce periodic yaw oscillations that the bandpass filter could misclassify as head shaking.

**Option A (recommended)**: **Dedicated Head Pose Estimator (6DRepNet)** — Replace L2CS-Net gaze vectors with [6DRepNet](https://github.com/thohemp/6DRepNet) head pose angles (pitch, yaw, roll). 6DRepNet outputs head orientation independent of eye gaze direction, which is precisely what nodding/shaking detection requires. Run 6DRepNet on the same bystander crops that 03a already processes, and add `head_pitch_rad`/`head_yaw_rad` fields to the 03a attention trace (additive schema change).
  - *Pros*: Eliminates eye saccade false positives; 6DRepNet is MIT-licensed and ~50MB; purpose-built for head orientation; can coexist alongside L2CS-Net (gaze for attention, head pose for gestures).
  - *Cons*: Requires modifying Layer 03a to run a second model per frame; adds ~20ms/frame inference time; increases 03a's complexity and memory footprint.

**Option B**: **Gaze-to-Head Proxy Filtering** — Keep L2CS-Net gaze vectors but apply a **low-pass filter** before the bandpass. Head movements are inherently slower and larger-amplitude than eye saccades. A 0.5Hz low-pass filter on the raw pitch/yaw would suppress fast saccadic oscillations while preserving genuine head nods (1-3Hz range after low-pass pre-filtering).
  - *Pros*: Zero additional model dependencies; trivial to implement (add one `scipy.signal.butter` call); no schema changes.
  - *Cons*: Heuristic — cannot reliably distinguish all saccades from small head nods; may suppress fast but genuine nods; adds a tuning parameter.

Your selection: Proceed with Option A.

---

### Issue 2: Nyquist Ceiling on Fast Nods
**Status**: ⚠️ Confirmed Unresolved — Layer 03a's adaptive stride samples at ~2.5-5 FPS. By the Nyquist-Shannon theorem, the maximum detectable frequency at 2.5 FPS is 1.25 Hz. Standard communicative nods occur at 1.5-3 Hz, meaning a significant portion of genuine nods are physically undetectable at the current sampling rate. The three-tier Nyquist-aware bandpass strategy mitigates filter design issues but cannot recover information that was never sampled.

**Option A (recommended)**: **Targeted High-FPS Sub-Sampling in 03e** — When 03e detects that the attention trace sampling rate is below 6 Hz (Nyquist for 3 Hz nods), re-read the source video within the reaction window at 10 FPS, running only L2CS-Net (or 6DRepNet per Issue 1) on the bystander crops. This creates a localized high-resolution pitch/yaw trace for gesture detection only, without requiring 03a to increase its global sampling rate.
  - *Pros*: Decouples gesture temporal resolution from attention temporal resolution; only runs the high-FPS pass when needed; 03a's output schema is unchanged.
  - *Cons*: Breaks 03e's "no re-inference" design principle; adds video file I/O to a layer that was previously pure signal processing; increases per-video processing time by ~5-10s.

**Option B**: **Increase 03a's Default Sampling Rate** — Change 03a's default stride from 0.5s (2 FPS) to 0.2s (5 FPS) globally. At 5 FPS, the Nyquist ceiling is 2.5 Hz, which captures most communicative nods.
  - *Pros*: Fixes the root cause for all downstream consumers; simple configuration change; no architectural complexity.
  - *Cons*: 2.5x more L2CS-Net inference calls for every video; increases 03a processing time proportionally; may not be necessary for videos without gesture-relevant tasks.

Your selection: Proceed with Option B but change the default stride to 8 FPS.

---

### Issue 3: Confidence Formula Energy Weighting
**Status**: ⚠️ Confirmed Unresolved — The current `confidence` metric is based solely on the count of zero-crossings (peaks and troughs) detected by `find_peaks`. A vigorous, large-amplitude nod and a barely-perceptible micro-nod receive the same confidence score as long as they have the same number of oscillation cycles. This makes it impossible for downstream consumers (e.g., the Emotion Corollary in Section 3) to distinguish emphatic agreement from ambiguous head movement.

**Option A (recommended)**: **RMS Amplitude Weighting** — Compute the RMS (root mean square) amplitude of the bandpass-filtered signal within the reaction window. Multiply the current count-based confidence by a normalized RMS factor: `confidence = count_confidence * min(1.0, rms / rms_threshold)`. Set `rms_threshold` to the median RMS observed across a calibration batch (e.g., 0.05 radians for nods).
  - *Pros*: Distinguishes emphatic from subtle gestures; single multiplicative factor; preserves existing count-based logic.
  - *Cons*: Requires calibrating `rms_threshold` on real data; RMS is sensitive to signal noise (pre-filtering quality matters).

Your selection: Proceed with Option A.

---

### Issue 4: Misleading Field Semantics (`*_variance_hz`)
**Status**: ⚠️ Confirmed Unresolved — The output schema fields `pitch_variance_hz` and `yaw_variance_hz` contain oscillation frequencies (e.g., 2.1 Hz), not statistical variance measures. The name `variance_hz` is semantically incorrect and could mislead downstream consumers (e.g., a researcher might expect these to represent frequency-domain variance, not the dominant oscillation frequency).

**Option A (recommended)**: **Rename to `*_oscillation_hz` in Next Schema Version** — Update the field names to `pitch_oscillation_hz` and `yaw_oscillation_hz` in the output schema. Since the schema follows an additive-only policy, add the new fields alongside the old ones (with identical values) for one version cycle, then deprecate the old names.
  - *Pros*: Semantically correct; eliminates confusion for downstream consumers; follows the additive-only schema rule.
  - *Cons*: Temporary field duplication during the deprecation cycle; requires updating all downstream consumers (03b Emotion Corollary, 04 export) to use the new field names.

Your selection: Proceed with Option A.

### 🧪 Test Suite Results (6/6 Passed)

A comprehensive verification suite in `tests/test_layer_03e.py` validates the following:
- **Nod/Shake/None Classification:** Synthetic sine-wave verification at ~2Hz.
- **NaN Resilience:** Gap patching using `np.interp` before filtration.
- **Hard Dependency Validation:** Raising `RuntimeError` on missing dependency files.
- **Ambiguous Wobble:** Correct classification of simultaneous equal-amplitude oscillations.
- **Non-Uniform Sampling:** Verification of detection accuracy after uniform resampling.
