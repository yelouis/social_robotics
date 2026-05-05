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

## ⚠️ Unresolved Issues & Suggestions

*None at this time.*

### 🧪 Test Suite Results (6/6 Passed)

A comprehensive verification suite in `tests/test_layer_03e.py` validates the following:
- **Nod/Shake/None Classification:** Synthetic sine-wave verification at ~2Hz.
- **NaN Resilience:** Gap patching using `np.interp` before filtration.
- **Hard Dependency Validation:** Raising `RuntimeError` on missing dependency files.
- **Ambiguous Wobble:** Correct classification of simultaneous equal-amplitude oscillations.
- **Non-Uniform Sampling:** Verification of detection accuracy after uniform resampling.
