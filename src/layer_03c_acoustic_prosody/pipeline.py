import os
import json
import logging
import tempfile
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.layer_03c_acoustic_prosody.config import Layer03cConfig

logger = logging.getLogger(__name__)

# Bracketed event tokens emitted by SenseVoice for non-speech audio events.
# Bare-word matching (e.g. "laughter" in transcribed text) was removed because
# transcribed speech routinely contains words like "slaughter", "laughtered",
# "applauded", "coughed", "crying", etc., which would falsely trigger events.
SENSEVOICE_EVENT_TOKENS = {
    "<|laughter|>": "laughter",
    "<|applause|>": "applause",
    "<|cough|>": "cough",
    "<|crying|>": "crying",
    "<|sneeze|>": "sneeze",
}


class AcousticProsodyPipeline:
    def __init__(
        self,
        input_manifest_path: str,
        output_result_path: str,
        force: bool = False,
        config: Layer03cConfig = Layer03cConfig(),
    ):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.error_log_path = self.output_result_path.parent / "03c_acoustic_prosody_errors.json"
        self.force = force
        self.config = config
        self.processed_ids = set()

        self._load_processed_ids()
        self._init_models()

    # SenseVoice (~150-200 MB resident) is eager-loaded on hosts with >= 48 GB
    # unified memory so the low-confidence gate never pays its ~2-3s load
    # latency mid-run. Smaller hosts (e.g. the 24 GB Mac mini M4 Pro) stay
    # lazy. Mirrors AttentionLayerPipeline.HIGH_MEMORY_HOST_BYTES (03a).
    HIGH_MEMORY_HOST_BYTES = 48 * 2**30

    @staticmethod
    def host_can_eager_load_sensevoice():
        """True when the host has enough unified memory to keep SenseVoice
        resident for the whole run. Falls back to False (lazy load) when
        psutil is not installed."""
        try:
            import psutil
        except ImportError:
            return False
        return psutil.virtual_memory().total >= AcousticProsodyPipeline.HIGH_MEMORY_HOST_BYTES

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
        # SenseVoice is invoked only when emotion2vec+ dominant confidence
        # falls below the gating threshold. On high-memory hosts (>= 48 GB
        # unified memory) it is eager-loaded below so the ambiguous-audio
        # clips that trigger the gate do not pay its ~2-3s load latency
        # mid-run; on smaller hosts it stays lazy-loaded (see
        # _run_sensevoice_model) to avoid ~150-200 MB of resident memory it
        # may never use.
        self.sensevoice_model = None
        try:
            from funasr import AutoModel
            self._AutoModel = AutoModel
            logger.info("Initializing emotion2vec+ via FunASR...")
            # disable_update=True prevents downloading updates on every run if already cached
            self.model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)
            self.funasr_available = True
            logger.info("emotion2vec+ initialized successfully.")
        except ImportError:
            raise RuntimeError(
                "funasr is not installed. Speech Emotion Recognition requires it. "
                "Install with: venv/bin/pip install funasr torchaudio"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load funasr SER model: {e}. "
                "Verify the ModelScope cache at ~/.cache/modelscope or re-run with "
                "disable_update=False to force re-download."
            ) from e

        # Eager-load SenseVoice on high-memory hosts. This is a pure
        # optimization: if construction fails, fall back to the lazy path,
        # which retries on first use and degrades to empty audio_events if
        # still unavailable — SenseVoice is supplementary, so unlike the
        # primary SER model its load failure does not warrant a hard fail.
        if self.host_can_eager_load_sensevoice():
            try:
                logger.info("High-memory host detected; eager-loading SenseVoiceSmall...")
                self.sensevoice_model = self._AutoModel(
                    model="iic/SenseVoiceSmall", disable_update=True
                )
                logger.info("SenseVoiceSmall eager-loaded successfully.")
            except Exception as e:
                logger.error(
                    f"Eager-load of SenseVoiceSmall failed ({e}); falling back to "
                    "lazy load on first low-confidence sample."
                )
                self.sensevoice_model = None
        else:
            logger.info("Standard-memory host; SenseVoice deferred until first low-confidence sample.")

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
                pitch_contour_variance = min(
                    1.0, pitch_variance / self.config.pitch_variance_normalization
                )
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
        """Run SenseVoiceSmall to detect non-speech audio events (laughter, applause, etc.).

        Lazy-loads the SenseVoice model on first invocation when the host did
        not eager-load it at init time (see host_can_eager_load_sensevoice).
        Matches only the bracketed event tokens emitted by SenseVoice's
        special-symbol grammar
        (e.g. ``<|laughter|>``); bare-word matching was retired because it
        false-positived on transcribed speech (``slaughter``, ``coughed``,
        ``the laughter died down``, etc.).
        """
        if not self.funasr_available:
            return []

        if self.sensevoice_model is None:
            try:
                logger.info("Lazy-loading SenseVoiceSmall on first low-confidence sample...")
                self.sensevoice_model = self._AutoModel(
                    model="iic/SenseVoiceSmall", disable_update=True
                )
            except Exception as e:
                logger.error(f"Failed to lazy-load SenseVoiceSmall: {e}")
                return []

        try:
            res = self.sensevoice_model.generate(wav_path, output_dir=None, disable_pbar=True)
            if not res or len(res) == 0:
                return []

            prediction = res[0]
            text_lower = prediction.get("text", "").lower()

            return [
                label for token, label in SENSEVOICE_EVENT_TOKENS.items()
                if token in text_lower
            ]
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
        
        cfg = self.config
        # Volume heuristic — cutoffs for "high" vs "low" volume relative to dBFS.
        is_high_volume = max_amp_dbFS > cfg.high_volume_dbfs
        is_low_volume = max_amp_dbFS < cfg.low_volume_dbfs

        alarming_score = angry + fearful + (cfg.high_volume_bonus if is_high_volume else 0.0)
        soothing_score = happy + surprised + (pitch_variance * cfg.pitch_variance_soothing_weight)
        negative_score = sad + (cfg.low_volume_bonus if is_low_volume else 0.0)
        
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
        if dominant_tone[1] < cfg.min_dominant_tone_score and tone_class != "Neutral":
            tone_class = "Neutral"
            
        return tone_class, scalars[tone_class]

    def _process_task(self, video_path: str, task: Dict) -> Optional[Dict]:
        task_id = task.get("task_id")
        temporal = task.get("task_temporal_metadata", {})
        window = temporal.get("task_reaction_window_sec")

        if not window or len(window) != 2:
            return None

        start_sec, end_sec = window

        # wav_path is initialized to None so the finally cleanup is always safe
        # to call regardless of which branch raised.
        wav_path = None
        try:
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
            if dominant_confidence < self.config.sensevoice_confidence_threshold:
                audio_events = self._run_sensevoice_model(wav_path)

            tone, scalar = self._classify_acoustic_tone(emotions, max_amp, pitch_var)

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
        finally:
            # Always clean up the extracted wav file, even on librosa/funasr/SenseVoice
            # failures that propagate up to run()'s outer try/except.
            if wav_path is not None:
                self._safe_remove(wav_path)

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
