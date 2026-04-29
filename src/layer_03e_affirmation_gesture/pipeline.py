import json
import traceback
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d

class AffirmationGesturePipeline:
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
                if result:
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

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))
        
        att_entry = self.attention_data.get(video_id)
        if not att_entry:
            print(f"No attention data found for {video_id}, skipping.")
            return None
            
        tasks = entry.get('identified_tasks', [])
        if not tasks:
            print(f"No tasks found for {video_id}.")
            return None
            
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
                
                if len(window_trace) < 5:  # Too few points to filter
                    continue
                
                # Extract time, pitch, yaw
                timestamps = np.array([x['t'] for x in window_trace])
                pitch = np.array([x['pitch_rad'] for x in window_trace])
                yaw = np.array([x['yaw_rad'] for x in window_trace])
                
                # Interpolate missing/NaN
                pitch = self._fill_nan(pitch)
                yaw = self._fill_nan(yaw)
                
                if len(pitch) == 0 or len(yaw) == 0:
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
                n_uniform = max(5, int(duration * fps))
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
                
                # Tie-breaking: if both axes oscillate similarly, classify as ambiguous
                if nod_confidence > 0.6 and shake_confidence > 0.6 and abs(nod_confidence - shake_confidence) < 0.15:
                    gesture = "ambiguous_wobble"
                    conf = round((nod_confidence + shake_confidence) / 2.0, 2)
                elif nod_confidence > 0.6 and nod_confidence > shake_confidence:
                    gesture = "affirming_nod"
                    conf = nod_confidence
                elif shake_confidence > 0.6:
                    gesture = "negating_shake"
                    conf = shake_confidence
                    
                per_person_results.append({
                    "person_id": person_id,
                    "pitch_variance_hz": round(pitch_var, 2),
                    "yaw_variance_hz": round(yaw_var, 2),
                    "gesture_detected": gesture,
                    "confidence": round(conf, 2)
                })
                
            if per_person_results:
                tasks_analyzed.append({
                    "task_id": task_id,
                    "per_person": per_person_results
                })
                
        if not tasks_analyzed:
            return None
            
        return {
            "video_id": video_id,
            "layer": "03e_affirmation_gesture",
            "tasks_analyzed": tasks_analyzed
        }

    def _fill_nan(self, arr):
        mask = np.isnan(arr)
        if mask.all():
            return np.zeros_like(arr)
        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
        return arr

    def _detect_oscillation(self, signal, fps, axis='pitch'):
        # Rhythmic nodding/shaking: typically 1Hz to 3Hz
        # We need a nyquist frequency. fps might be around 2-5fps based on 03a adaptive stride
        nyq = 0.5 * fps
        
        # If fps is too low to capture 1-3Hz, we can't properly filter.
        # But we can still count zero-crossings on the raw detrended signal.
        signal_detrended = signal - np.mean(signal)
        
        # Attempt bandpass if nyquist allows a meaningful passband, otherwise
        # fall back to raw detrended signal with peak-finding only.
        # A "meaningful" band requires the normalized passband width to be > 0.1.
        if nyq > 3.0:
            # Nyquist comfortably above target range: standard 1-3Hz bandpass
            low = 1.0 / nyq
            high = 3.0 / nyq
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

        # Find peaks and troughs (oscillating zero-crossings)
        # We need >2 crossings or peaks
        # For a nod, we want noticeable variance.
        std_dev = np.std(filtered)
        if std_dev < 0.02: # Too small, barely moving
            return 0.0, 0.0
            
        # Find peaks with some prominence
        peaks, _ = find_peaks(filtered, prominence=0.03)
        troughs, _ = find_peaks(-filtered, prominence=0.03)
        
        total_extrema = len(peaks) + len(troughs)
        
        # We approximate frequency from the number of extrema
        duration = len(signal) / fps
        est_hz = (total_extrema / 2.0) / duration if duration > 0 else 0
        
        confidence = 0.0
        # If we have at least 2 distinct motions (e.g. peak + trough + peak) in a short window
        if total_extrema >= 2 and 0.5 <= est_hz <= 4.0:
            confidence = min(1.0, 0.5 + (total_extrema * 0.1))
            
        return confidence, float(est_hz)
