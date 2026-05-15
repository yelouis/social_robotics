import json
import traceback
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.interpolate import interp1d

class AffirmationGesturePipeline:
    # --- Detection Tuning ---
    MIN_TRACE_POINTS = 5
    MEDFILT_KERNEL = 3
    MAX_INTERPOLATED_FRACTION = 0.3
    PEAK_PROMINENCE = 0.03
    STD_DEV_FLOOR = 0.02
    FREQ_BAND_HZ = (0.5, 4.0)
    BANDPASS_HZ = (1.0, 3.0)
    COUNT_CONFIDENCE_BASE = 0.5
    COUNT_CONFIDENCE_PER_EXTREMUM = 0.1
    RMS_THRESHOLD = 0.05
    GESTURE_DECISION_THRESHOLD = 0.6
    AMBIGUITY_DELTA = 0.15
    # Engage the medfilt saccade suppressor at 03a's M4-Max 32 FPS burst stride.
    # 3-sample window at 32 FPS = 94 ms, comfortably below the 167 ms half-period
    # of a 3 Hz nod so the filter does not demolish genuine fast nods.
    MEDFILT_MIN_FPS = 32.0
    # Per-gaze-model interpolation thresholds (Resolved Issue 16). The default
    # 0.3 was calibrated against L2CS-Net's tracking-loss profile; alternative
    # upstream gaze models report different tracking-loss distributions, so the
    # threshold is re-keyed at startup from 03a's processing_meta.model_used.
    INTERPOLATION_THRESHOLDS = {
        "l2cs_net_3d_gaze": 0.3,
        "crossgaze": 0.2,
        "3dgazenet": 0.25,
    }

    def __init__(self, input_manifest_path, output_result_path, attention_result_path, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.attention_result_path = Path(attention_result_path)
        self.error_log_path = self.output_result_path.parent / "03e_affirmation_gesture_errors.json"
        self.force = force
        
        self.processed_ids = set()
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting fresh.")

        # HARD DEPENDENCY: 03e requires 03a's attention_trace data.
        # Per 03_social_layer_architecture.md, this is a required cross-layer input.
        self.attention_data = {}
        if not self.attention_result_path.exists():
            raise RuntimeError(
                f"HARD DEPENDENCY FAILURE: 03a_attention_result.json not found at "
                f"{self.attention_result_path}. Layer 03e cannot function without "
                f"Layer 03a's attention trace data. Please run Layer 03a first."
            )
        try:
            with open(self.attention_result_path, 'r') as f:
                att_results = json.load(f)
                for r in att_results:
                    self.attention_data[r['video_id']] = r
            if not self.attention_data:
                raise RuntimeError(
                    "03a_attention_result.json is empty. Layer 03e requires at least "
                    "one processed video from Layer 03a."
                )
            print(f"Loaded attention data for {len(self.attention_data)} videos from 03a.")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse 03a_attention_result.json: {e}")

        # Re-key the interpolation threshold against 03a's reported gaze model.
        # The default (class-level) 0.3 holds for L2CS-Net and unknown models.
        active_model = self._detect_upstream_gaze_model(att_results)
        if active_model in self.INTERPOLATION_THRESHOLDS:
            self.MAX_INTERPOLATED_FRACTION = self.INTERPOLATION_THRESHOLDS[active_model]
            print(
                f"Calibrated MAX_INTERPOLATED_FRACTION to "
                f"{self.MAX_INTERPOLATED_FRACTION} for upstream gaze model "
                f"'{active_model}'."
            )

    @staticmethod
    def _detect_upstream_gaze_model(att_results):
        for r in att_results:
            model = r.get('processing_meta', {}).get('model_used')
            if model and model != 'fallback_dummy':
                return model
        return None

    def _log_error(self, video_id, error):
        error_entry = {
            "video_id": video_id,
            "error": str(error),
            "traceback": traceback.format_exc()
        }
        print(f"Error processing {video_id}: {error}")
        
        errors = []
        if self.error_log_path.exists():
            try:
                with open(self.error_log_path, 'r') as f:
                    errors = json.load(f)
            except:
                pass
        
        errors.append(error_entry)
        
        temp_err = self.error_log_path.with_suffix('.tmp')
        with open(temp_err, 'w') as f:
            json.dump(errors, f, indent=4)
        temp_err.replace(self.error_log_path)

    def run(self):
        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)

        results = []
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    results = json.load(f)
            except:
                pass

        for entry in registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue

            print(f"Processing Affirmation Gesture for video: {video_id}")
            try:
                result = self.process_video(entry)
                results.append(result)
                self.processed_ids.add(video_id)

                # Atomic write
                temp_out = self.output_result_path.with_suffix('.tmp')
                with open(temp_out, 'w') as f:
                    json.dump(results, f, indent=4)
                temp_out.replace(self.output_result_path)
            except Exception as e:
                self._log_error(video_id, e)

        print(f"Final count: {len(results)} videos processed for Affirmation Gesture.")

    def _sentinel(self, video_id, reason):
        return {
            "video_id": video_id,
            "layer": "03e_affirmation_gesture",
            "tasks_analyzed": [],
            "skipped_reason": reason,
        }

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))

        att_entry = self.attention_data.get(video_id)
        if not att_entry:
            print(f"No attention data found for {video_id}, skipping.")
            return self._sentinel(video_id, "no_attention_data")

        tasks = entry.get('identified_tasks', [])
        if not tasks:
            print(f"No tasks found for {video_id}.")
            return self._sentinel(video_id, "no_tasks")

        tasks_analyzed = []
        for task in tasks:
            task_id = task.get('task_id', 'unknown')
            meta = task.get('task_temporal_metadata', {})
            reaction_window = meta.get('task_reaction_window_sec')

            if not reaction_window or len(reaction_window) != 2:
                continue

            start_sec, end_sec = reaction_window
            per_person_att = att_entry.get('per_person', [])

            per_person_results = []
            for p_data in per_person_att:
                person_id = p_data.get('person_id')
                trace = p_data.get('attention_trace', [])

                # Filter trace by window
                window_trace = [t for t in trace if start_sec <= t['t'] <= end_sec]

                if len(window_trace) < self.MIN_TRACE_POINTS:
                    continue

                # Extract time, pitch, yaw
                timestamps = np.array([x['t'] for x in window_trace])
                pitch = np.array([x['pitch_rad'] for x in window_trace])
                yaw = np.array([x['yaw_rad'] for x in window_trace])

                # Interpolate NaNs and capture how much of each axis was bridged.
                # If too much of the trace was reconstructed, downstream filtering
                # produces fabricated oscillations from the interpolant ramp, so
                # short-circuit the bystander.
                pitch, pitch_interp_frac = self._fill_nan(pitch)
                yaw, yaw_interp_frac = self._fill_nan(yaw)
                interpolated_fraction = max(pitch_interp_frac, yaw_interp_frac)

                if len(pitch) == 0 or len(yaw) == 0:
                    continue
                if interpolated_fraction > self.MAX_INTERPOLATED_FRACTION:
                    continue

                # Resample to a uniform grid before filtering.
                # 03a uses adaptive stride (0.2s or 0.5s), but scipy.signal.filtfilt
                # assumes uniform sampling. We interpolate onto a fixed-dt grid.
                duration = timestamps[-1] - timestamps[0]
                if duration <= 0:
                    continue
                dt = np.mean(np.diff(timestamps))
                if dt <= 0:
                    continue
                fps = 1.0 / dt
                n_uniform = max(self.MIN_TRACE_POINTS, int(duration * fps))
                t_uniform = np.linspace(timestamps[0], timestamps[-1], n_uniform)

                pitch_interp = interp1d(timestamps, pitch, kind='linear', fill_value='extrapolate')
                yaw_interp = interp1d(timestamps, yaw, kind='linear', fill_value='extrapolate')
                pitch_uniform = pitch_interp(t_uniform)
                yaw_uniform = yaw_interp(t_uniform)

                uniform_fps = (n_uniform - 1) / duration if duration > 0 else fps

                # Detect nod/shake
                nod_confidence, pitch_var = self._detect_oscillation(pitch_uniform, uniform_fps, axis='pitch')
                shake_confidence, yaw_var = self._detect_oscillation(yaw_uniform, uniform_fps, axis='yaw')

                gesture = "none"
                conf = 0.0

                threshold = self.GESTURE_DECISION_THRESHOLD
                # Tie-breaking: if both axes oscillate similarly, classify as ambiguous
                if (nod_confidence > threshold and shake_confidence > threshold
                        and abs(nod_confidence - shake_confidence) < self.AMBIGUITY_DELTA):
                    gesture = "ambiguous_wobble"
                    conf = round((nod_confidence + shake_confidence) / 2.0, 2)
                elif nod_confidence > threshold and nod_confidence > shake_confidence:
                    gesture = "affirming_nod"
                    conf = nod_confidence
                elif shake_confidence > threshold:
                    gesture = "negating_shake"
                    conf = shake_confidence

                per_person_results.append({
                    "person_id": person_id,
                    "pitch_oscillation_hz": round(pitch_var, 2),
                    "yaw_oscillation_hz": round(yaw_var, 2),
                    "interpolated_fraction": round(interpolated_fraction, 3),
                    "gesture_detected": gesture,
                    "confidence": round(conf, 2),
                })

            if per_person_results:
                tasks_analyzed.append({
                    "task_id": task_id,
                    "per_person": per_person_results
                })

        if not tasks_analyzed:
            return self._sentinel(video_id, "insufficient_trace")

        return {
            "video_id": video_id,
            "layer": "03e_affirmation_gesture",
            "tasks_analyzed": tasks_analyzed
        }

    def _fill_nan(self, arr):
        mask = np.isnan(arr)
        if len(arr) == 0:
            return arr, 0.0
        if mask.all():
            return np.zeros_like(arr), 1.0
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
        return arr, float(mask.sum()) / float(len(arr))

    def _detect_oscillation(self, signal, fps, axis='pitch'):
        # Rhythmic nodding/shaking: typically 1Hz to 3Hz.
        nyq = 0.5 * fps

        # Median-smoothing saccade suppression (Resolved Issue 10).
        # `medfilt` rejects single-sample impulses (saccades) but distorts pure
        # tones whose period approaches the kernel duration. Only fire when the
        # sampling rate is high enough that kernel*dt is well below the longest
        # nod half-period (1/(2*FREQ_BAND_HZ.high)).
        bp_low_hz, bp_high_hz = self.BANDPASS_HZ
        if fps >= self.MEDFILT_MIN_FPS and len(signal) >= self.MEDFILT_KERNEL:
            signal = medfilt(signal, kernel_size=self.MEDFILT_KERNEL)

        signal_detrended = signal - np.mean(signal)

        # Attempt bandpass if nyquist allows a meaningful passband, otherwise
        # fall back to raw detrended signal with peak-finding only.
        if nyq > bp_high_hz:
            # Nyquist comfortably above target range: standard bandpass
            low = bp_low_hz / nyq
            high = bp_high_hz / nyq
            b, a = butter(2, [low, high], btype='band')
            filtered = filtfilt(b, a, signal_detrended)
        elif nyq > 1.5:
            # Nyquist is marginal: use widest possible band
            low = max(0.01, 0.5 / nyq)
            high = 0.99
            if high - low > 0.1:
                b, a = butter(2, [low, high], btype='band')
                filtered = filtfilt(b, a, signal_detrended)
            else:
                filtered = signal_detrended
        else:
            # Nyquist too low for any filtering — use raw detrended signal
            filtered = signal_detrended

        std_dev = np.std(filtered)
        if std_dev < self.STD_DEV_FLOOR:
            return 0.0, 0.0

        peaks, _ = find_peaks(filtered, prominence=self.PEAK_PROMINENCE)
        troughs, _ = find_peaks(-filtered, prominence=self.PEAK_PROMINENCE)

        total_extrema = len(peaks) + len(troughs)

        duration = len(signal) / fps
        est_hz = (total_extrema / 2.0) / duration if duration > 0 else 0

        confidence = 0.0
        freq_lo, freq_hi = self.FREQ_BAND_HZ
        if total_extrema >= 2 and freq_lo <= est_hz <= freq_hi:
            count_confidence = min(
                1.0,
                self.COUNT_CONFIDENCE_BASE + (total_extrema * self.COUNT_CONFIDENCE_PER_EXTREMUM),
            )
            rms = np.sqrt(np.mean(filtered ** 2))
            confidence = count_confidence * min(1.0, rms / self.RMS_THRESHOLD)

        return confidence, float(est_hz)
