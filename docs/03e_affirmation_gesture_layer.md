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

## 🧪 Resolved Issues & Implementation Refinements (April 2026)

1. **Bandpass Cutoff at Low FPS (Resolved)**:
   - **Problem**: Layer 03a uses an adaptive stride (e.g., dropping to ~2.5-5 FPS when static), resulting in a low Nyquist frequency (e.g., ~2.4 Hz). If the SciPy bandpass filter cutoff is statically set to `1.5 Hz` for low FPS, it mathematically erases the `2.0 Hz` signature of a standard head nod, leading to false negatives (`none` detected).
   - **Solution**: Implemented a three-tier Nyquist-aware bandpass strategy: full 1-3Hz band when `nyq > 3.0`, widest-possible band (`0.5/nyq` to `0.99`) when `nyq > 1.5`, and raw detrended fallback for extremely low sampling rates.

2. **Non-Uniform Sampling Breaks `filtfilt` (Resolved)**:
   - **Problem**: `scipy.signal.filtfilt` assumes uniform sampling, but 03a's adaptive stride produces intervals varying from 0.2s to 0.5s (a 2.5x ratio). Using `np.mean(np.diff(timestamps))` as the sampling rate produces incorrect frequency response.
   - **Solution**: Added a uniform resampling step using `scipy.interpolate.interp1d` before filtering. The pipeline now interpolates pitch/yaw onto a fixed-dt grid derived from the mean sampling rate, ensuring `filtfilt` operates on correctly-spaced samples.

3. **Missing 03a Dependency Fails Silently (Resolved)**:
   - **Problem**: Per `03_social_layer_architecture.md`, Layer 03e has a **hard dependency** on 03a's `attention_trace`. If the attention result file was missing or empty, the pipeline silently skipped all videos with "No attention data found" instead of raising an error. This violated the documented hard dependency contract and could waste compute time on long batch runs with zero output.
   - **Solution**: The `__init__` method now raises `RuntimeError` immediately if `03a_attention_result.json` does not exist, is empty, or fails to parse. This ensures the hard dependency is validated before any processing begins.

4. **Simultaneous Nod+Shake Misclassified (Resolved)**:
   - **Problem**: When both pitch and yaw oscillated with equal confidence (e.g., a diagonal head wobble), the pipeline arbitrarily classified the gesture based on the `elif` branch order rather than recognizing the ambiguity.
   - **Solution**: Added a tie-breaking check: if both `nod_confidence` and `shake_confidence` exceed 0.6 and are within 0.15 of each other, the gesture is classified as `ambiguous_wobble` instead of forcing an arbitrary nod/shake label.

## ⚠️ Known Errors & Limitations

1. **Gaze Vectors vs. Head Pose Vectors Conflation (Design-Level)**:
   - **Problem**: Layer 03e consumes `pitch_rad` and `yaw_rad` from 03a, which are L2CS-Net **gaze direction** vectors (where the eyes are looking), not **head pose** angles (head orientation). Gaze direction includes rapid eye saccades and micro-movements that are faster and noisier than actual head movements. A genuine head nod produces a pitch oscillation of ~0.1-0.3 rad, but gaze saccades can produce similar magnitudes without any head movement, potentially causing false positives.
   - **Impact**: The layer may incorrectly flag rapid eye scanning patterns (e.g., reading, looking around a room) as head nods or shakes.
   - **Recommended Solution**: Replace the L2CS-Net gaze vectors with a dedicated **head pose estimator** (e.g., `6DRepNet`, `WHENet`, or MediaPipe Face Mesh head pose). This would require either (a) modifying Layer 03a to also output head pose alongside gaze, or (b) adding a lightweight head pose inference step directly in 03e. Option (a) is preferred to avoid re-opening videos.

2. **Nyquist Ceiling on Detectable Nod Frequency (Fundamental)**:
   - **Problem**: Layer 03a's adaptive stride produces an effective sampling rate of ~2-5 FPS (mean ~2.5 FPS in the slow-movement regime). By the Nyquist theorem, this means the maximum detectable oscillation frequency is ~1.0-1.25 Hz. Fast head nods at 2-3 Hz are **physically undetectable** from this data and will be aliased into lower frequencies, producing incorrect classifications.
   - **Impact**: Only slow, deliberate nods (~1 Hz) are reliably detected. Rapid communicative head gestures are missed.
   - **Recommended Solution**: Increase 03a's sampling density during detected social interaction windows. Alternatively, 03e could perform its own targeted high-fps (10+ FPS) sub-sampling of the raw video for reaction windows where a gesture is suspected, trading compute for accuracy.

3. **Confidence Formula Is Count-Based, Not Amplitude-Weighted (Low)**:
   - **Problem**: The confidence formula `min(1.0, 0.5 + total_extrema * 0.1)` is purely based on the count of detected peaks/troughs. It does not factor in the amplitude or energy of the oscillation. This means that many tiny, barely-visible wiggles could theoretically produce the same confidence as a few large, dramatic nods.
   - **Impact**: Low in practice because the `std_dev < 0.02` gate filters out most micro-movements. However, borderline cases in the 0.02-0.05 rad range may produce inflated confidence scores.
   - **Recommended Solution**: Incorporate the RMS (root mean square) amplitude of the filtered signal as a weighting factor: `confidence = min(1.0, (0.5 + total_extrema * 0.1) * min(1.0, rms / 0.05))`.

4. **Field Name Semantics: `*_variance_hz` (Cosmetic)**:
   - **Problem**: The output fields `pitch_variance_hz` and `yaw_variance_hz` contain the **estimated oscillation frequency** in Hz, not a statistical variance measure. The naming follows the original design doc schema but is semantically misleading.
   - **Impact**: Downstream consumers may misinterpret the field as a variance metric rather than a frequency estimate.
   - **Recommended Solution**: If the output schema is updated in a future version, rename to `pitch_oscillation_hz` / `yaw_oscillation_hz`.

### 🧪 Test Suite Results (6/6 Passed)

A comprehensive verification suite was built in `tests/test_layer_03e.py`:
- **Nod/Shake/None Classification:** Generates synthetic sine-wave oscillations at ~2Hz (at 5 FPS, well above Nyquist) to verify correct classification of `affirming_nod`, `negating_shake`, and `none`.
- **NaN Resilience:** Intentionally drops slices of frames to simulate L2CS-Net tracking failures, verifying that `np.interp` patches gaps before SciPy filtration.
- **Hard Dependency Validation:** Asserts that `RuntimeError` is raised when the attention file is missing or empty.
- **Ambiguous Wobble:** Verifies that simultaneous equal-amplitude oscillation on both axes is classified as `ambiguous_wobble`.
- **Non-Uniform Sampling:** Tests with 03a's realistic adaptive stride (alternating 0.2s/0.5s intervals) at 1Hz nod frequency, confirming correct detection after uniform resampling.
