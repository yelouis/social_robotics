# AI Task Breakdown: Affirmation Gesture Layer (03e)

## Objective
The **Affirmation Gesture Layer** parses the most explicit non-verbal heuristic an infant uses for moral validation: Head Nodding and Head Shaking. Even if an adult is smiling, a shaking head signals "No." This layer extracts rhythmic, high-frequency spatial oscillations from the bystander's head tracking data to explicitly classify affirmation or negation.

---

## 📥 Input Requirements
- **`03a_attention_result.json`** (required cross-layer): This layer acts as a direct mathematical extension of 03a. We reuse the 3D Pitch/Yaw vectors extracted by L2CS-Net during the attention phase. The pipeline strictly requires this file and must fail-fast (`RuntimeError`) if missing or empty, to prevent silent failures and wasted compute.
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

### 🧪 Test Suite Results (7/7 Passed)

A comprehensive verification suite in `tests/test_layer_03e.py` validates the following:
- **Nod/Shake/None Classification:** Synthetic sine-wave verification at ~2Hz.
- **NaN Resilience:** Gap patching using `np.interp` before filtration.
- **Hard Dependency Validation:** Raising `RuntimeError` on missing dependency files.
- **Ambiguous Wobble:** Correct classification of simultaneous equal-amplitude oscillations.
- **Non-Uniform Sampling:** Verification of detection accuracy after uniform resampling.
- **Upstream Gaze-Model Calibration:** `MAX_INTERPOLATED_FRACTION` re-keys to 0.3 / 0.2 / 0.25 against `processing_meta.model_used` across L2CS-Net, CrossGaze, 3DGazeNet, unknown-model, fallback-only, and missing-meta branches.
