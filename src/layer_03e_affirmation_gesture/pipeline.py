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
    # Issue 2 (June 14): 03e read-window anchoring. 03a only runs gaze inference
    # within +/-WINDOW_PAD_SEC of each bystander DETECTION, so the strict
    # task_reaction_window_sec (wearer-climax +/-1s) is usually empty (bystander
    # detections sit a median ~6.8s from the climax — docs/03d Resolved #1).
    # Mirror 03d's _bystander_measurement_window: keep the reaction window when it
    # already holds >= MIN_TRACE_POINTS samples, else anchor to the climax-nearest
    # detection +/- ANCHOR_SPAN_DETECTIONS (padded by WINDOW_PAD_SEC), capped at
    # MAX_ANCHOR_SPAN_SEC so a sparse track can't stretch the window to minutes.
    ANCHOR_SPAN_DETECTIONS = 1
    WINDOW_PAD_SEC = 2.0
    MAX_ANCHOR_SPAN_SEC = 30.0
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
            if entry.get("synthetic") is True:
                continue
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

        # Issue 2 (June 14): map person_id -> bystander DETECTION timestamps so a
        # person whose trace is absent from the strict reaction window can re-anchor
        # to where 03a actually sampled (around the detections).
        det_by_pid = {b.get('person_id'): b.get('timestamps_sec', [])
                      for b in entry.get('bystander_detections', [])}

        tasks_analyzed = []
        skip_reasons = []
        for task in tasks:
            task_id = task.get('task_id', 'unknown')
            meta = task.get('task_temporal_metadata', {})
            reaction_window = meta.get('task_reaction_window_sec')

            if not reaction_window or len(reaction_window) != 2:
                continue

            start_sec, end_sec = reaction_window
            climax_sec = meta.get('task_climax_sec', (start_sec + end_sec) / 2.0)
            per_person_att = att_entry.get('per_person', [])

            per_person_results = []
            for p_data in per_person_att:
                # Issue 3 (June 14): isolate per-person failures. A single short
                # trace used to raise out of filtfilt and abort the WHOLE clip
                # (every other person lost, no record emitted). Contain it here.
                try:
                    person_id = p_data.get('person_id')
                    trace = p_data.get('attention_trace', [])
                    trace_ts = [x['t'] for x in trace]

                    # Issue 2: align the read window with where 03a actually sampled.
                    w_start, w_end, w_source = self._measurement_window(
                        trace_ts, det_by_pid.get(person_id, []), climax_sec, start_sec, end_sec)
                    if w_start is None:
                        skip_reasons.append(w_source)
                        continue

                    window_trace = [t for t in trace if w_start <= t['t'] <= w_end]
                    if len(window_trace) < self.MIN_TRACE_POINTS:
                        skip_reasons.append("insufficient_trace")
                        continue

                    # Extract time, pitch, yaw
                    timestamps = np.array([x['t'] for x in window_trace])
                    # Issue 2 (June 14 post-fix finding): 03a writes lost-tracking
                    # samples as pitch=yaw=0.0 with target='NoFace' (NOT NaN), so the
                    # step transitions in/out of those zero-plateaus fabricate "nods"
                    # and interpolated_fraction never flags them. Treat NoFace as the
                    # missing data it is -> NaN, so it flows through _fill_nan + the
                    # interpolated_fraction guard instead of forming real-valued steps.
                    # docs/03e Issue 1 (Option A): prefer true HEAD POSE
                    # (head_pitch_rad/head_yaw_rad from 03a's FaceLandmarker) over
                    # gaze — gaze moves with eye saccades while the head is still.
                    # A sample's head pose is null when FaceLandmarker found no face
                    # (its own tracking-loss) or the sample was NoFace, so treat
                    # null as missing (NaN) exactly like the gaze NoFace handling.
                    if any(x.get('head_pitch_rad') is not None for x in window_trace):
                        signal_source = 'head_pose'
                        pitch = np.array([x['head_pitch_rad'] if x.get('head_pitch_rad') is not None else np.nan for x in window_trace])
                        yaw = np.array([x['head_yaw_rad'] if x.get('head_yaw_rad') is not None else np.nan for x in window_trace])
                    else:
                        signal_source = 'gaze'
                        pitch = np.array([np.nan if x.get('target') == 'NoFace' else x['pitch_rad'] for x in window_trace])
                        yaw = np.array([np.nan if x.get('target') == 'NoFace' else x['yaw_rad'] for x in window_trace])

                    # Interpolate NaNs and capture how much of each axis was bridged.
                    # If too much of the trace was reconstructed, downstream filtering
                    # produces fabricated oscillations from the interpolant ramp, so
                    # short-circuit the bystander.
                    pitch, pitch_interp_frac = self._fill_nan(pitch)
                    yaw, yaw_interp_frac = self._fill_nan(yaw)
                    interpolated_fraction = max(pitch_interp_frac, yaw_interp_frac)

                    if len(pitch) == 0 or len(yaw) == 0:
                        skip_reasons.append("insufficient_trace")
                        continue
                    if interpolated_fraction > self.MAX_INTERPOLATED_FRACTION:
                        skip_reasons.append("over_interpolated")
                        continue

                    # Resample to a uniform grid before filtering.
                    # 03a uses adaptive stride (0.2s or 0.5s), but scipy.signal.filtfilt
                    # assumes uniform sampling. We interpolate onto a fixed-dt grid.
                    duration = timestamps[-1] - timestamps[0]
                    if duration <= 0:
                        skip_reasons.append("insufficient_trace")
                        continue
                    dt = np.mean(np.diff(timestamps))
                    if dt <= 0:
                        skip_reasons.append("insufficient_trace")
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
                        "measurement_window_sec": [round(float(w_start), 2), round(float(w_end), 2)],
                        "window_source": w_source,
                        "signal_source": signal_source,
                    })
                except Exception as e:
                    # One bystander's failure must not poison the rest of the clip.
                    print(f"  per-person {p_data.get('person_id')} failed for {video_id}: {e}")
                    skip_reasons.append("error")
                    continue

            if per_person_results:
                tasks_analyzed.append({
                    "task_id": task_id,
                    "per_person": per_person_results
                })

        if not tasks_analyzed:
            return self._sentinel(video_id, self._aggregate_skip_reason(skip_reasons))

        return {
            "video_id": video_id,
            "layer": "03e_affirmation_gesture",
            "tasks_analyzed": tasks_analyzed
        }

    def _measurement_window(self, trace_timestamps, det_timestamps, climax_sec, start_sec, end_sec):
        """Issue 2 (June 14): align 03e's read window with where 03a sampled.

        03a only runs gaze inference within +/-WINDOW_PAD_SEC of each bystander
        DETECTION, so the strict task_reaction_window_sec (wearer-climax +/-1s)
        usually contains zero trace points (detections sit a median ~6.8s from
        the climax — docs/03d Resolved #1), which scored the layer 0/50. Mirrors
        03d's _bystander_measurement_window: keep the reaction window when it
        already holds >= MIN_TRACE_POINTS trace samples; otherwise anchor to the
        DETECTION nearest the climax +/- ANCHOR_SPAN_DETECTIONS, padded by
        WINDOW_PAD_SEC (03a's sampling pad) so the dense trace around those
        detections is captured, and bounded by MAX_ANCHOR_SPAN_SEC.

        Returns (start, end, source); on skip (None, None, reason) where reason
        is 'insufficient_trace' (no usable detection to anchor on) or
        'span_capped' (anchored span exceeds the cap).
        """
        in_win = sum(1 for t in trace_timestamps if start_sec <= t <= end_sec)
        if in_win >= self.MIN_TRACE_POINTS:
            return start_sec, end_sec, "reaction_window"
        det = sorted(det_timestamps)
        if not det:
            return None, None, "insufficient_trace"
        i = min(range(len(det)), key=lambda k: abs(det[k] - climax_sec))
        lo = max(0, i - self.ANCHOR_SPAN_DETECTIONS)
        hi = min(len(det) - 1, i + self.ANCHOR_SPAN_DETECTIONS)
        ws = det[lo] - self.WINDOW_PAD_SEC
        we = det[hi] + self.WINDOW_PAD_SEC
        if (we - ws) > self.MAX_ANCHOR_SPAN_SEC:
            return None, None, "span_capped"
        return ws, we, "bystander_anchored"

    @staticmethod
    def _aggregate_skip_reason(reasons):
        """Collapse per-person skip reasons into one sentinel reason (Issue 2)."""
        if not reasons:
            return "insufficient_trace"
        return reasons[0] if len(set(reasons)) == 1 else "mixed_skip"

    @staticmethod
    def _safe_filtfilt(b, a, sig):
        """Issue 3 (June 14): filtfilt RAISES when len(sig) <= padlen
        (3*max(len(a),len(b)) = 15 for a 2nd-order Butterworth bandpass) instead
        of degrading. Guard it and signal the caller to fall back to the raw
        detrended signal + peak-finding rather than crash the whole clip."""
        padlen = 3 * max(len(a), len(b))
        if len(sig) <= padlen:
            return None
        return filtfilt(b, a, sig)

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
            ff = self._safe_filtfilt(b, a, signal_detrended)
            filtered = ff if ff is not None else signal_detrended
        elif nyq > 1.5:
            # Nyquist is marginal: use widest possible band
            low = max(0.01, 0.5 / nyq)
            high = 0.99
            if high - low > 0.1:
                b, a = butter(2, [low, high], btype='band')
                ff = self._safe_filtfilt(b, a, signal_detrended)
                filtered = ff if ff is not None else signal_detrended
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
