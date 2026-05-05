import os
import json
import logging
import tempfile
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class AcousticProsodyPipeline:
    def __init__(
        self,
        input_manifest_path: str,
        output_result_path: str,
        force: bool = False
    ):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03c_acoustic_prosody_errors.json"
        self.force = force
        self.processed_ids = set()

        self._load_processed_ids()
        self._init_models()

    def _init_models(self):
        """Initialize the funasr SER model, librosa, and validate ffmpeg.
        
        Raises RuntimeError with actionable install instructions if any
        required dependency is missing. This prevents the pipeline from
        silently producing all-Neutral results when dependencies are absent.
        """
        self.model = None
        self.funasr_available = False
        self.librosa_available = False
        self.ffmpeg_available = False

        # --- FFmpeg validation ---
        import shutil
        if shutil.which("ffmpeg"):
            self.ffmpeg_available = True
        else:
            raise RuntimeError(
                "ffmpeg is not installed or not on PATH. "
                "Audio slicing requires ffmpeg. Install with: brew install ffmpeg"
            )

        # --- Librosa validation ---
        try:
            import librosa
            self.librosa_available = True
            self.librosa = librosa
        except ImportError:
            raise RuntimeError(
                "librosa is not installed. Acoustic feature extraction requires it. "
                "Install with: venv/bin/pip install librosa"
            )

        # --- FunASR + emotion2vec+ validation ---
        try:
            from funasr import AutoModel
            logger.info("Initializing emotion2vec+ via FunASR...")
            # disable_update=True prevents downloading updates on every run if already cached
            self.model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)
            logger.info("Initializing SenseVoiceSmall via FunASR...")
            self.sensevoice_model = AutoModel(model="iic/SenseVoiceSmall", disable_update=True)
            self.funasr_available = True
            logger.info("emotion2vec+ and SenseVoiceSmall initialized successfully.")
        except ImportError:
            raise RuntimeError(
                "funasr is not installed. Speech Emotion Recognition requires it. "
                "Install with: venv/bin/pip install funasr torchaudio"
            )
        except Exception as e:
            logger.error(f"Failed to load funasr model: {e}")

    def _load_processed_ids(self):
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    results = json.load(f)
                    for r in results:
                        self.processed_ids.add(r.get("video_id"))
            except Exception as e:
                logger.error(f"Error reading existing results: {e}")

    def _extract_audio_chunk(self, video_path: str, start_sec: float, end_sec: float) -> Optional[str]:
        """Extract a chunk of audio from the video using ffmpeg and save it to a temporary 16kHz wav file.
        
        Uses tempfile.mkstemp for process-safe unique filenames to avoid collisions
        when multiple pipeline instances run concurrently.
        """
        if start_sec >= end_sec:
            logger.warning(f"Invalid time window: start={start_sec}, end={end_sec}")
            return None
            
        # Use mkstemp for a guaranteed-unique, process-safe temporary file
        fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix="prosody_")
        os.close(fd)  # Close the file descriptor; ffmpeg will write to the path
        
        duration = end_sec - start_sec
        # ffmpeg command to extract audio, resample to 16kHz, mono
        cmd = [
            "ffmpeg",
            "-y", # Overwrite
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(duration),
            "-vn", # Disable video
            "-acodec", "pcm_s16le", # 16-bit PCM
            "-ar", "16000", # 16kHz sample rate
            "-ac", "1", # Mono
            out_wav
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(out_wav) and os.path.getsize(out_wav) > 44:
                # WAV header is 44 bytes; anything <= 44 means no actual audio data
                return out_wav
            else:
                logger.warning(f"Extracted audio file is empty or missing: {out_wav}")
                self._safe_remove(out_wav)
                return None
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed to extract audio from {video_path}: {e}")
            self._safe_remove(out_wav)
            return None
        except FileNotFoundError:
            logger.error("ffmpeg is not installed on the system.")
            self._safe_remove(out_wav)
            return None

    def _extract_librosa_features(self, wav_path: str) -> Tuple[float, float]:
        """Extract max amplitude and pitch variance using librosa."""
        if not self.librosa_available:
            return 0.0, 0.0
            
        try:
            y, sr = self.librosa.load(wav_path, sr=16000)
            if len(y) == 0:
                return 0.0, 0.0
                
            # Max amplitude in dBFS
            max_amp = float(np.max(np.abs(y)))
            max_amp_dbFS = float(20 * np.log10(max_amp + 1e-6))
            
            # Pitch contour variance using pyin.
            # pyin returns (f0, voiced_flag, voiced_probs).
            # f0 contains NaN for unvoiced frames — filter with ~np.isnan(f0).
            f0, voiced_flag, voiced_probs = self.librosa.pyin(
                y,
                fmin=self.librosa.note_to_hz('C2'),
                fmax=self.librosa.note_to_hz('C7'),
                sr=sr
            )
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 1:
                pitch_variance = float(np.var(valid_f0))
                # Normalize variance to [0, 1] range for heuristic consumption
                pitch_contour_variance = min(1.0, pitch_variance / 10000.0) 
            else:
                pitch_contour_variance = 0.0
                
            return max_amp_dbFS, pitch_contour_variance
        except Exception as e:
            logger.error(f"Error extracting librosa features for {wav_path}: {e}")
            return 0.0, 0.0

    def _run_ser_model(self, wav_path: str) -> Dict[str, float]:
        """Run funasr emotion2vec+ model.
        
        emotion2vec+ returns labels in this fixed order:
        0: angry, 1: disgusted, 2: fearful, 3: happy, 4: neutral,
        5: other, 6: sad, 7: surprised, 8: unknown
        """
        # The expected 9 classes
        default_scores = {
            "angry": 0.0, "happy": 0.0, "sad": 0.0,
            "surprised": 0.0, "fearful": 0.0, "neutral": 1.0,
            "disgusted": 0.0, "other": 0.0, "unknown": 0.0
        }
        
        if not self.funasr_available or not self.model:
            return default_scores
            
        try:
            # AutoModel.generate returns a list of dicts with 'labels' and 'scores' keys.
            res = self.model.generate(wav_path, output_dir=None, disable_pbar=True)
            if not res or len(res) == 0:
                return default_scores
                
            prediction = res[0]
            labels = prediction.get("labels", [])
            scores = prediction.get("scores", [])
            
            emotion_scores = {}
            for label, score in zip(labels, scores):
                # Handle 'Chinese/English' format labels (e.g. '生气/angry')
                clean_label = label.lower().strip()
                if '/' in clean_label:
                    clean_label = clean_label.split('/')[-1].strip()
                
                if clean_label in default_scores:
                    emotion_scores[clean_label] = float(score)
            
            # Fill missing with 0.0
            for k in default_scores:
                if k not in emotion_scores:
                    emotion_scores[k] = 0.0
                    
            return emotion_scores
        except Exception as e:
            logger.error(f"Error running SER model on {wav_path}: {e}")
            return default_scores

    def _run_sensevoice_model(self, wav_path: str) -> List[str]:
        """Run SenseVoiceSmall to detect non-speech audio events (laughter, applause, etc.)."""
        if not self.funasr_available or getattr(self, "sensevoice_model", None) is None:
            return []
            
        try:
            res = self.sensevoice_model.generate(wav_path, output_dir=None, disable_pbar=True)
            if not res or len(res) == 0:
                return []
                
            prediction = res[0]
            text = prediction.get("text", "")
            text_lower = text.lower()
            
            events = []
            if "laughter" in text_lower or "<|laughter|>" in text_lower:
                events.append("laughter")
            if "applause" in text_lower or "<|applause|>" in text_lower:
                events.append("applause")
            if "cough" in text_lower or "<|cough|>" in text_lower:
                events.append("cough")
            if "crying" in text_lower or "<|crying|>" in text_lower:
                events.append("crying")
            if "sneeze" in text_lower or "<|sneeze|>" in text_lower:
                events.append("sneeze")
                
            return events
        except Exception as e:
            logger.error(f"Error running SenseVoice model on {wav_path}: {e}")
            return []

    def _classify_acoustic_tone(self, emotion_scores: Dict[str, float], max_amp_dbFS: float, pitch_variance: float) -> Tuple[str, float]:
        """
        Heuristic Mapping:
        - Alarming / Deterrent: High angry + high fearful + Sudden Volume Spike
        - Soothing / Encouraging: High happy + high surprised + Melodic Pitch Contour
        - Discouraging / Negative: High sad + Low Volume
        """
        angry = emotion_scores.get("angry", 0.0)
        fearful = emotion_scores.get("fearful", 0.0)
        happy = emotion_scores.get("happy", 0.0)
        surprised = emotion_scores.get("surprised", 0.0)
        sad = emotion_scores.get("sad", 0.0)
        
        # Volume heuristic (arbitrary threshold for "high" vs "low" volume if baseline is unknown, say -20 dBFS)
        is_high_volume = max_amp_dbFS > -20.0
        is_low_volume = max_amp_dbFS < -35.0
        
        alarming_score = angry + fearful + (0.3 if is_high_volume else 0.0)
        soothing_score = happy + surprised + (pitch_variance * 0.5)
        negative_score = sad + (0.3 if is_low_volume else 0.0)
        
        scores = {
            "Alarming": alarming_score,
            "Soothing": soothing_score,
            "Discouraging": negative_score,
            "Neutral": emotion_scores.get("neutral", 0.0)
        }
        
        dominant_tone = max(scores.items(), key=lambda x: x[1])
        
        # Scalar mapping: Alarming (-1.0), Discouraging (-0.5), Neutral (0.0), Soothing (+1.0)
        scalars = {
            "Alarming": -1.0,
            "Discouraging": -0.5,
            "Neutral": 0.0,
            "Soothing": 1.0
        }
        
        tone_class = dominant_tone[0]
        # If confidence is very low, default to Neutral
        if dominant_tone[1] < 0.3 and tone_class != "Neutral":
            tone_class = "Neutral"
            
        return tone_class, scalars[tone_class]

    def _process_task(self, video_path: str, task: Dict) -> Optional[Dict]:
        task_id = task.get("task_id")
        temporal = task.get("task_temporal_metadata", {})
        window = temporal.get("task_reaction_window_sec")
        
        if not window or len(window) != 2:
            return None
            
        start_sec, end_sec = window
        
        wav_path = self._extract_audio_chunk(video_path, start_sec, end_sec)
        if not wav_path:
            # If no audio or extraction failed, return a neutral stub
            return {
                "task_id": task_id,
                "task_reaction_window_sec": window,
                "prosody_metrics": {
                    "max_amplitude_dbFS": -100.0,
                    "pitch_contour_variance": 0.0,
                    "emotion_scores": {k: (1.0 if k=="neutral" else 0.0) for k in ["angry", "happy", "sad", "surprised", "fearful", "neutral", "disgusted", "other", "unknown"]},
                    "dominant_emotion": "neutral",
                    "dominant_emotion_confidence": 1.0,
                    "audio_events": []
                },
                "classified_acoustic_tone": "Neutral",
                "prosody_scalar": 0.0
            }

        max_amp, pitch_var = self._extract_librosa_features(wav_path)
        emotions = self._run_ser_model(wav_path)
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        dominant_confidence = dominant_emotion[1]
        
        audio_events = []
        if dominant_confidence < 0.6:
            audio_events = self._run_sensevoice_model(wav_path)
        
        tone, scalar = self._classify_acoustic_tone(emotions, max_amp, pitch_var)
        
        # Clean up temp file
        self._safe_remove(wav_path)
            
        return {
            "task_id": task_id,
            "task_reaction_window_sec": window,
            "prosody_metrics": {
                "max_amplitude_dbFS": max_amp,
                "pitch_contour_variance": pitch_var,
                "emotion_scores": emotions,
                "dominant_emotion": dominant_emotion[0],
                "dominant_emotion_confidence": dominant_confidence,
                "audio_events": audio_events
            },
            "classified_acoustic_tone": tone,
            "prosody_scalar": scalar
        }

    def run(self):
        if not self.input_manifest_path.exists():
            logger.error(f"Manifest not found: {self.input_manifest_path}")
            return
            
        with open(self.input_manifest_path, 'r') as f:
            manifest = json.load(f)
            
        results = []
        if self.output_result_path.exists() and not self.force:
            with open(self.output_result_path, 'r') as f:
                results = json.load(f)
                
        for video_data in manifest:
            video_id = video_data.get("video_id")
            if video_id in self.processed_ids and not self.force:
                continue
                
            video_path = video_data.get("video_path")
            if not video_path or not os.path.exists(video_path):
                logger.warning(f"Video missing or invalid path: {video_path}")
                continue
                
            identified_tasks = video_data.get("identified_tasks", [])
            tasks_analyzed = []
            
            try:
                for task in identified_tasks:
                    task_res = self._process_task(video_path, task)
                    if task_res:
                        tasks_analyzed.append(task_res)
            except Exception as e:
                self._log_error(video_id, e)
                continue
                    
            if tasks_analyzed:
                res_record = {
                    "video_id": video_id,
                    "layer": "03c_acoustic_prosody",
                    "tasks_analyzed": tasks_analyzed
                }
                results.append(res_record)
                self.processed_ids.add(video_id)
                
                # Atomic write per video to ensure resumability
                self._write_results(results)

    def _write_results(self, results: List[Dict]):
        temp_path = self.output_result_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(results, f, indent=2)
        temp_path.replace(self.output_result_path)

    def _log_error(self, video_id: str, error: Exception):
        """Log per-video errors to a dedicated error file (per 03 architecture policy)."""
        error_entry = {
            "video_id": video_id,
            "error": str(error),
            "traceback": traceback.format_exc()
        }
        logger.error(f"Error processing {video_id}: {error}")

        errors = []
        if self.error_log_path.exists():
            try:
                with open(self.error_log_path, 'r') as f:
                    errors = json.load(f)
            except Exception:
                pass

        errors.append(error_entry)
        with open(self.error_log_path, 'w') as f:
            json.dump(errors, f, indent=4)

    @staticmethod
    def _safe_remove(path: str):
        """Remove a file, ignoring errors if it doesn't exist."""
        try:
            os.remove(path)
        except OSError:
            pass

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Run Acoustic Prosody Layer 03c")
    parser.add_argument("--manifest", default="filtered_manifest.json", help="Path to input filtered_manifest.json")
    parser.add_argument("--output", default="03c_acoustic_prosody_result.json", help="Path to output result JSON")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    args = parser.parse_args()
    
    pipeline = AcousticProsodyPipeline(args.manifest, args.output, args.force)
    pipeline.run()
