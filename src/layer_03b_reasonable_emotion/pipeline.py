import json
import cv2
import traceback
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Literal, Optional

import numpy as np

# Emotion model backend (Resolved Issue #4). The torch HSEmotion checkpoint
# fails to load under torch>=2.6 (weights_only) + CUDA-pickle + timm-version
# mismatch, so we PREFER the ONNX backend (hsemotion-onnx) — same
# predict_emotions API, ONNX weights, no torch/timm pickle. Fall back to the
# torch backend, then to None (which triggers a LOUD mock warning at init).
HSEMOTION_BACKEND = None
try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    HSEMOTION_BACKEND = "onnx"
except ImportError:
    try:
        from hsemotion.facial_emotions import HSEmotionRecognizer
        HSEMOTION_BACKEND = "torch"
    except ImportError:
        HSEmotionRecognizer = None

# Optional dependency: torch — used only for MPS device detection
try:
    import torch
except ImportError:
    torch = None

# Face-presence gate (Resolved Issue #3, mirrors 03a Resolved Issue #2).
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None

import os as _os
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FACE_DETECTOR_PATH = _PROJECT_ROOT / "models" / "mediapipe" / "blaze_face_short_range.tflite"
MIN_FACE_CONF = float(_os.getenv("SR_03B_MIN_FACE_CONF", "0.5"))
ENABLE_FACE_GATE = _os.getenv("SR_03B_FACE_GATE", "1").lower() in ("1", "true", "yes")
# Emotion confidence gate (Resolved Issue 1, June 04). HSEmotion's 8-class
# softmax has a 0.125 uniform baseline; on distant/blurry egocentric crops (or a
# BlazeFace false positive) the top probability stays ~0.18 — the model is
# guessing. Drop any emotion sample below this top-probability threshold so a
# task with no confident sample scores nothing (honest) instead of emitting
# noise. Default 0.4 sits just above the ~0.38 noise ceiling observed on Ego4D.
MIN_EMOTION_CONF = float(_os.getenv("SR_03B_MIN_EMOTION_CONF", "0.4"))

# Optional dependency: Ollama (kept for the fast local `.list()` availability
# probe only — the actual chat calls go through the enforced-timeout httpx path).
try:
    import ollama as ollama_client
except ImportError:
    ollama_client = None

# Enforced-timeout HTTP path for the LLM calls. ollama-python ignores its own
# client timeout (sends timeout=None per request), so a wedged generation would
# hang the unattended batch — the same failure 02's VLM gate hit (docs/02
# Resolved #20). vlm_client.ollama_chat posts to /api/chat via httpx with a real
# timeout that closes the socket and cancels the generation on expiry.
try:
    from shared.vlm_client import ollama_chat
except ImportError:
    from src.shared.vlm_client import ollama_chat

# ---------------------------------------------------------------------------
#  Pydantic schemas for LLM output validation (Limitation #2 fix)
# ---------------------------------------------------------------------------

class ExpectationSchema(BaseModel):
    """Step 1 output: expected emotions per outcome polarity."""
    positive_emotions: List[str]
    negative_emotions: List[str]
    neutral_baseline: List[str]

    @field_validator("positive_emotions", "negative_emotions", "neutral_baseline", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class TransitionEvalSchema(BaseModel):
    """Step 3 output: classification of a single emotion transition."""
    classified_direction: Literal["positive", "negative", "neutral"]
    reasoning: str = ""

# HSEmotion model + label mapping. The 8-class HSEmotion models emit the
# FER+ vocabulary; we fold it onto the canonical 03b label set the rest of
# the layer consumes. `contempt` has no 03b analog and is folded onto
# `disgust` (closest negative-valence label); `happiness` is renamed to `joy`.
try:
    from src.models_config import get_model
except ImportError:
    from models_config import get_model
HSEMOTION_MODEL = get_model("layer_03b_face_emotion")
HSEMOTION_TO_CANONICAL = {
    "anger": "anger",
    "contempt": "disgust",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}

# ---------------------------------------------------------------------------
#  Default / fallback expectations (used when LLM is unavailable)
# ---------------------------------------------------------------------------
DEFAULT_EXPECTATIONS = {
    "positive_emotions": ["joy", "amusement", "relief"],
    "negative_emotions": ["anger", "disgust", "fear", "sadness"],
    "neutral_baseline": ["neutral", "surprise"]
}

# LLM configuration. The 03b reasoning chain (Step 1 expectation generation +
# Step 3 cumulative-history evaluation) is a multi-step structured-output regime.
# The tier-per-host config points `medium`/`large` at the installed `gemma4:latest`
# (~8B) — `gemma4:26b` was registered but never pulled on this host (Resolved
# Issue #9); re-pull a 26B-class tag here for higher reasoning fidelity.
OLLAMA_MODEL = get_model("layer_03b_ollama")
LLM_MAX_RETRIES = 3
# Generous per-call bound (JSON reasoning runs longer than a one-word gate);
# enforced via httpx in vlm_client.ollama_chat so a stuck generation can't hang
# the batch. On timeout the call raises and the retry/fallback path handles it.
LLM_REQUEST_TIMEOUT = 300


class ReasonableEmotionPipeline:
    def __init__(self, input_manifest_path, output_result_path, attention_result_path=None, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.attention_result_path = Path(attention_result_path) if attention_result_path else None
        self.error_log_path = self.output_result_path.parent / "03b_reasonable_emotion_errors.json"
        self.force = force

        # Initialize the face-emotion model — HSEmotion-PyTorch, which runs
        # natively on the M4 Max MPS GPU. Instantiated once and reused across
        # all videos to avoid per-frame model-loading cost. Falls back to CPU
        # when MPS is unavailable, and to the deterministic mock generator in
        # _sample_emotions when the package itself is absent.
        self.emotion_device = "mps" if (torch is not None and torch.backends.mps.is_available()) else "cpu"
        self.emotion_detector = None
        self.emotion_source = "mock"
        try:
            if HSEmotionRecognizer and HSEMOTION_BACKEND == "onnx":
                # ONNX backend takes only model_name (onnxruntime picks the provider).
                self.emotion_detector = HSEmotionRecognizer(model_name=HSEMOTION_MODEL)
                self.emotion_source = "hsemotion_onnx"
                print(f"[03b] HSEmotion ONNX backend loaded ({HSEMOTION_MODEL}).")
            elif HSEmotionRecognizer and HSEMOTION_BACKEND == "torch":
                self.emotion_detector = HSEmotionRecognizer(
                    model_name=HSEMOTION_MODEL, device=self.emotion_device
                )
                self.emotion_source = "hsemotion_torch"
                print(f"[03b] HSEmotion torch backend loaded ({self.emotion_device} mode).")
        except Exception as e:
            print(f"[03b] Failed to load HSEmotion ({HSEMOTION_BACKEND}): {e}")
            self.emotion_detector = None
        if self.emotion_detector is None:
            # Resolved Issue #4: never run silently on mock emotions.
            print("[03b] *** WARNING: no real emotion model loaded — FALLING BACK TO "
                  "MOCK (seeded-RNG) EMOTIONS. Results are NOT real (emotion_source="
                  "'mock'). Install `hsemotion-onnx` to fix. ***")

        # Lazy MediaPipe BlazeFace face gate (Resolved Issue #3). Only used when a
        # real emotion model is present; the mock path needs no face.
        self._face_detector = None
        self.enable_face_gate = (
            ENABLE_FACE_GATE and mp is not None and FACE_DETECTOR_PATH.exists()
            and self.emotion_detector is not None
        )

        # Check Ollama availability once at init
        self.ollama_available = False
        if ollama_client:
            try:
                ollama_client.list()
                self.ollama_available = True
                print(f"[03b] Ollama available. Will use model '{OLLAMA_MODEL}'.")
            except Exception:
                print("[03b] Ollama not reachable. Will use fallback expectations.")

        self.processed_ids = set()
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    existing_data = json.load(f)
                    self.processed_ids = {entry['video_id'] for entry in existing_data}
                print(f"Resuming: {len(self.processed_ids)} videos already processed.")
            except Exception as e:
                print(f"Error loading existing results: {e}. Starting fresh.")

        self.attention_data = {}
        if self.attention_result_path and self.attention_result_path.exists():
            try:
                with open(self.attention_result_path, 'r') as f:
                    att_data = json.load(f)
                    for item in att_data:
                        self.attention_data[item['video_id']] = item
            except Exception as e:
                print(f"Error loading attention results: {e}")

    # ------------------------------------------------------------------
    #  Attention helper
    # ------------------------------------------------------------------
    def get_attention_score(self, video_id, person_id):
        if video_id in self.attention_data:
            for p in self.attention_data[video_id].get('per_person', []):
                if p.get('person_id') == person_id:
                    return p.get('average_attention_score', 1.0)
        return 1.0  # Default if no attention data

    # ------------------------------------------------------------------
    #  LLM helpers (Limitation #2 fix — pydantic + retry)
    # ------------------------------------------------------------------
    def _llm_generate_expectations(self, task_label: str) -> dict:
        """Step 1: Generate baseline emotional expectations via Gemma 4.

        Uses pydantic validation and retry logic.  Falls back to hardcoded
        defaults if the LLM is unavailable or returns invalid JSON after
        all retries.
        """
        if not self.ollama_available:
            return dict(DEFAULT_EXPECTATIONS)

        prompt = (
            f'You are analyzing bystander reactions to this task: "{task_label}"\n\n'
            "Generate sets of emotions a bystander would display if:\n"
            "1. The outcome is POSITIVE (successful, impressive).\n"
            "2. The outcome is NEGATIVE (failed, dangerous).\n\n"
            "Respond in EXACTLY this JSON format:\n"
            '{\n'
            '  "positive_emotions": ["emotion1", "emotion2"],\n'
            '  "negative_emotions": ["emotion1", "emotion2"],\n'
            '  "neutral_baseline": ["bored", "neutral", "distracted"]\n'
            '}'
        )

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                raw_text = ollama_chat(
                    OLLAMA_MODEL, prompt, fmt="json",
                    options={"temperature": 0.3 + (attempt - 1) * 0.2},
                    timeout=LLM_REQUEST_TIMEOUT,
                ).strip()
                parsed = json.loads(raw_text)
                validated = ExpectationSchema(**parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"[03b] Expectation LLM attempt {attempt}/{LLM_MAX_RETRIES} "
                      f"failed validation: {e}")
            except Exception as e:
                print(f"[03b] Expectation LLM attempt {attempt}/{LLM_MAX_RETRIES} "
                      f"error: {e}")

        print("[03b] All LLM retries exhausted for expectations. Using defaults.")
        return dict(DEFAULT_EXPECTATIONS)

    def _llm_evaluate_transition(self, task_label: str, expectations: dict,
                                  accumulated_history: str,
                                  emotion_start: str, emotion_end: str) -> dict:
        """Step 3: Evaluate a single emotion transition via Gemma 4.

        Uses pydantic validation and retry logic.  Falls back to rule-based
        classification if the LLM is unavailable or fails all retries.
        """
        if not self.ollama_available:
            return self._rule_based_classify(emotion_end, expectations)

        prompt = (
            f'Task: "{task_label}"\n'
            f'Predicted Positive Emotions: {expectations["positive_emotions"]}\n'
            f'Predicted Negative Emotions: {expectations["negative_emotions"]}\n\n'
            f'Previous Reaction History (for context):\n{accumulated_history}\n\n'
            f'Current Transition being Evaluated:\n'
            f'During the climax of the task, the bystander\'s emotion transitioned '
            f'explicitly from {emotion_start} to {emotion_end}.\n\n'
            f'Considering the history, does this new transition indicate a positive '
            f'or negative task execution?\n\n'
            f'Respond in EXACTLY this JSON format:\n'
            '{\n'
            '  "classified_direction": "positive" | "negative" | "neutral",\n'
            '  "reasoning": "Briefly map this specific transition to the task outcome context."\n'
            '}'
        )

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                raw_text = ollama_chat(
                    OLLAMA_MODEL, prompt, fmt="json",
                    options={"temperature": 0.3 + (attempt - 1) * 0.2},
                    timeout=LLM_REQUEST_TIMEOUT,
                ).strip()
                parsed = json.loads(raw_text)
                validated = TransitionEvalSchema(**parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"[03b] Transition eval attempt {attempt}/{LLM_MAX_RETRIES} "
                      f"failed validation: {e}")
            except Exception as e:
                print(f"[03b] Transition eval attempt {attempt}/{LLM_MAX_RETRIES} "
                      f"error: {e}")

        print("[03b] All LLM retries exhausted for transition eval. Using rule-based fallback.")
        return self._rule_based_classify(emotion_end, expectations)

    @staticmethod
    def _rule_based_classify(emotion: str, expectations: dict) -> dict:
        """Deterministic fallback: classify by simple set membership."""
        if emotion in expectations.get("positive_emotions", []):
            direction = "positive"
        elif emotion in expectations.get("negative_emotions", []):
            direction = "negative"
        else:
            direction = "neutral"
        return {"classified_direction": direction, "reasoning": "rule-based fallback"}

    # ------------------------------------------------------------------
    #  Main run loop
    # ------------------------------------------------------------------
    def run(self):
        # Resolved Issue #8: bystander-face-quality pre-filter. 03b is low-yield
        # on egocentric data (Resolved Issue #7) and climax optical-flow is the
        # dominant per-clip cost, so a cheap pre-pass (~0.9 s/clip) annotates each
        # clip with `bystander_face_quality` and the gate skips clips with no
        # close, resolvable face *before* paying climax + emotion. Idempotent and
        # env-toggleable (SR_FACE_QUALITY_GATE=0). Defensive: any failure (import
        # or scoring) leaves the gate open (every clip processed) so it can never
        # silently drop work.
        passes_face_quality = lambda entry: True  # noqa: E731 — fail-open default
        try:
            try:
                from shared.face_quality_prefilter import (
                    populate_face_quality_for_manifest, passes_face_quality)
            except ImportError:
                from src.shared.face_quality_prefilter import (
                    populate_face_quality_for_manifest, passes_face_quality)
            nq = populate_face_quality_for_manifest(self.input_manifest_path)
            if nq:
                print(f"[03b] Scored bystander face quality for {nq} manifest entries.")
        except Exception as e:
            print(f"[03b] face-quality pre-filter skipped ({e}); gate fails open.")
            passes_face_quality = lambda entry: True  # noqa: E731 — fail open

        # Resolved Issue #5: 03b is the consumer that needs the reaction window,
        # so it populates climax metadata itself (idempotent — a no-op once every
        # task has it), fulfilling the Layer-03 contract from 02 Resolved Issue #8.
        # The face-quality gate is passed as `entry_filter` so optical flow is
        # never paid on clips the gate will skip below.
        try:
            try:
                from shared.climax_extraction import populate_climax_for_manifest
            except ImportError:
                from src.shared.climax_extraction import populate_climax_for_manifest
            n = populate_climax_for_manifest(
                self.input_manifest_path, skip_vlm=True, entry_filter=passes_face_quality)
            if n:
                print(f"[03b] Populated climax metadata for {n} manifest entries.")
        except Exception as e:
            print(f"[03b] climax population skipped ({e}); proceeding with existing metadata.")

        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)

        results = []
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    results = json.load(f)
            except Exception:
                pass

        skipped_face_quality = 0
        for entry in registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
                continue

            # Resolved Issue #8: skip low-yield clips (no climax was computed for
            # them either, so _process_task would no-op anyway). Counted so the
            # gate's effect is visible in the run log.
            if not passes_face_quality(entry):
                skipped_face_quality += 1
                continue

            print(f"Processing Layer 03b for video: {video_id}")
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
                self.log_error(video_id, e)

        if skipped_face_quality:
            print(f"[03b] face-quality gate skipped {skipped_face_quality} low-yield clips "
                  f"(SR_FACE_QUALITY_GATE=0 to disable).")
        print(f"Final count: {len(results)} videos processed for Reasonable Emotion.")

    def log_error(self, video_id, error):
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
            except Exception:
                pass

        errors.append(error_entry)
        with open(self.error_log_path, 'w') as f:
            json.dump(errors, f, indent=4)

    def process_video(self, entry):
        video_id = entry.get('id', entry.get('video_id'))
        video_path = Path(entry['video_path'])

        if not video_path.exists():
            print(f"File not found: {video_path}")
            return None

        bystanders = entry.get('bystander_detections', [])
        if not bystanders:
            print(f"No bystanders found for {video_id}.")
            return None

        identified_tasks = entry.get('identified_tasks', [])
        if not identified_tasks:
            print(f"No clear tasks identified for {video_id}.")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            cap.release()
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0 or total_frames == 0:
            cap.release()
            return None

        tasks_analyzed = []
        # Bystander-aware multi-window climax: process one reaction segment per
        # bystander cluster (each pseudo-task carries a single-window metadata).
        try:
            from shared.climax_extraction import expand_task_segments
        except ImportError:
            from src.shared.climax_extraction import expand_task_segments
        # Cross-layer multi-window guardrail (docs/03 § Multi-Window Reaction
        # Segments): a sparse bystander's segments collapse onto the same
        # presence-anchored window, so score each distinct (person, window) once
        # per clip rather than re-counting the identical reaction per segment.
        scored_windows = set()
        for task in expand_task_segments(identified_tasks):
            task_result = self._process_task(cap, fps, task, bystanders, video_id, scored_windows)
            if task_result:
                # Segment attribution for the Layer-04 segment join (docs/06
                # Issue 2): which reaction segment this row measured.
                task_result["segment_index"] = task.get('segment_index', 0)
                tasks_analyzed.append(task_result)

        cap.release()

        if not tasks_analyzed:
            return None
        return {
            "video_id": video_id,
            "layer": "03b_reasonable_emotion",
            "emotion_source": self.emotion_source,
            "tasks_analyzed": tasks_analyzed
        }

    def _process_task(self, cap, fps, task, bystanders, video_id, scored_windows=None):
        task_id = task.get('task_id', 'unknown')
        task_label = task.get('task_label', '')
        task_temporal = task.get('task_temporal_metadata', {})
        window_sec = task_temporal.get('task_reaction_window_sec', [])
        climax_sec = task_temporal.get('task_climax_sec', 0.0)

        if not window_sec or len(window_sec) != 2:
            return None

        start_sec, end_sec = window_sec

        # Step 1: Expectation Generation — LLM with pydantic validation + retry
        expectations = self._llm_generate_expectations(task_label)

        # Step 2: single-pass emotion sampling (Resolved Issue #1) with
        # bystander-presence-anchored windows (Resolved Issue #2) and a face
        # gate (Resolved Issue #3). Returns {person_id: timeseries(>=2 samples)}.
        bystander_timeseries = self._collect_emotion_timeseries(
            cap, fps, task, bystanders, climax_sec, [start_sec, end_sec], scored_windows
        )

        per_person = []
        if bystander_timeseries:
            # Step 3: Run accumulated history evaluations in parallel across bystanders
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self._evaluate_bystander_transitions,
                        pid, ts, task_label, expectations, climax_sec
                    ): pid
                    for pid, ts in bystander_timeseries.items()
                }
                for future in futures:
                    try:
                        res = future.result()
                        if res and res.get("temporal_slices"):
                            per_person.append(res)
                    except Exception as e:
                        pid = futures[future]
                        print(f"[03b] Failed to evaluate transitions for person {pid}: {e}")

        if not per_person:
            return None

        # Task aggregate score — attention-weighted mean across bystanders.
        # Per the doc: "this final score is averaged across all bystander scores
        # for that task, weighted by their 03a attention span."
        attention_weights = []
        for p in per_person:
            att = self.get_attention_score(video_id, p['person_id'])
            attention_weights.append(att)

        weighted_sum = sum(
            p['late_stage_weighted_success_score'] * w
            for p, w in zip(per_person, attention_weights)
        )
        total_weight = sum(attention_weights)
        task_aggregate = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            "task_id": task_id,
            "task_label": task_label,
            "task_reaction_window_sec": window_sec,
            "per_person": per_person,
            "task_aggregate_score": round(task_aggregate, 2)
        }

    def _evaluate_bystander_transitions(self, person_id, timeseries, task_label, expectations, climax_sec):
        slices = []
        accumulated_history = ""
        for i in range(len(timeseries) - 1):
            start_emotion = timeseries[i]
            end_emotion = timeseries[i + 1]

            if start_emotion['emotion'] == end_emotion['emotion']:
                if slices:
                    slices[-1]['window_sec'][1] = end_emotion['t']
                    slices[-1]['terminal_magnitude'] = round(end_emotion['magnitude'], 2)
                    scalar = end_emotion['magnitude']
                    if slices[-1]['classified_direction'] == "negative":
                        scalar = -scalar
                    elif slices[-1]['classified_direction'] == "neutral":
                        scalar = 0.0
                    slices[-1]['slice_success_scalar'] = round(scalar, 2)
                else:
                    direction = self._rule_based_classify(end_emotion['emotion'], expectations)['classified_direction']
                    scalar = end_emotion['magnitude']
                    if direction == "negative":
                        scalar = -scalar
                    elif direction == "neutral":
                        scalar = 0.0
                    slices.append({
                        "slice_id": len(slices) + 1,
                        "window_sec": [start_emotion['t'], end_emotion['t']],
                        "transition_pair": [start_emotion['emotion'], end_emotion['emotion']],
                        "terminal_magnitude": round(end_emotion['magnitude'], 2),
                        "classified_direction": direction,
                        "slice_success_scalar": round(scalar, 2)
                    })
                continue

            # Use LLM with pydantic validation + retry, or rule-based fallback
            eval_result = self._llm_evaluate_transition(
                task_label, expectations, accumulated_history,
                start_emotion['emotion'], end_emotion['emotion']
            )
            direction = eval_result['classified_direction']

            # Build accumulated history string for next iteration
            accumulated_history += (
                f"{len(slices) + 1}. {start_emotion['emotion']} -> {end_emotion['emotion']} "
                f"(Classified: {direction})\n"
            )

            scalar = end_emotion['magnitude']
            if direction == "negative":
                scalar = -scalar
            elif direction == "neutral":
                scalar = 0.0

            slices.append({
                "slice_id": len(slices) + 1,
                "window_sec": [start_emotion['t'], end_emotion['t']],
                "transition_pair": [start_emotion['emotion'], end_emotion['emotion']],
                "terminal_magnitude": round(end_emotion['magnitude'], 2),
                "classified_direction": direction,
                "slice_success_scalar": round(scalar, 2)
            })

        # Calculate late-stage weighted success score
        numerator = 0.0
        denominator = 0.0
        for s in slices:
            s_val = s['slice_success_scalar']
            d_val = s['window_sec'][1] - s['window_sec'][0]
            t_start_i = s['window_sec'][0]
            w_val = max(0.1, t_start_i - climax_sec)
            numerator += (s_val * d_val * w_val)
            denominator += (d_val * w_val)

        person_score = 0.0
        if denominator > 0:
            person_score = numerator / denominator

        return {
            "person_id": person_id,
            "temporal_slices": slices,
            "late_stage_weighted_success_score": round(person_score, 2)
        }

    # ------------------------------------------------------------------
    #  Emotion sampling — single-pass, face-gated (Resolved Issues #1/#2/#3)
    # ------------------------------------------------------------------
    SAMPLE_DT = 0.33      # ~3 FPS within the reaction window
    MATCH_TOL = 2.0       # max gap (s) between a sample time and a bystander detection
    # Skip a bystander whose nearest detection is more than this from the segment
    # climax: it was not present at the task moment, so scoring it is wrong (not a
    # reaction to THIS task) and, on long clips with bystanders scattered across the
    # whole duration, it stretched 03b's single decode pass across the entire clip
    # (08a946f4: 192 bystanders over 0-5982s -> ~1.8M frame grabs). Mirrors 03d.
    MAX_ANCHOR_SPAN_SEC = 30.0

    def _face_detector_inst(self):
        if self._face_detector is None and self.enable_face_gate:
            base = mp_python.BaseOptions(model_asset_path=str(FACE_DETECTOR_PATH))
            opts = mp_vision.FaceDetectorOptions(
                base_options=base, running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=MIN_FACE_CONF,
            )
            self._face_detector = mp_vision.FaceDetector.create_from_options(opts)
        return self._face_detector

    def _extract_face(self, crop_bgr):
        """Resolved Issue #3: return the largest detected face sub-crop (BGR) from
        a bystander box, or None if no face is found. With the gate disabled
        (mock/no-model path) returns the whole crop so the mock path is unaffected."""
        if not self.enable_face_gate or crop_bgr.size == 0:
            return crop_bgr
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            det = self._face_detector_inst().detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not det.detections:
                return None
            d = max(det.detections, key=lambda x: x.bounding_box.width * x.bounding_box.height)
            bb = d.bounding_box
            H, W = crop_bgr.shape[:2]
            fx1, fy1 = max(0, bb.origin_x), max(0, bb.origin_y)
            fx2, fy2 = min(W, bb.origin_x + bb.width), min(H, bb.origin_y + bb.height)
            face = crop_bgr[fy1:fy2, fx1:fx2]
            return face if face.size > 0 else None
        except Exception:
            return None

    def _predict_emotions_batch(self, rgb_faces):
        """Resolved Issue #1: batched HSEmotion inference. Returns (labels, scores_list)."""
        try:
            labels, scores = self.emotion_detector.predict_multi_emotions(rgb_faces, logits=False)
            return list(labels), [scores[i] for i in range(len(rgb_faces))]
        except Exception:
            labels, out = [], []
            for f in rgb_faces:
                lab, sc = self.emotion_detector.predict_emotions(f, logits=False)
                labels.append(lab); out.append(sc)
            return labels, out

    @staticmethod
    def _mock_emotion(t, start_sec, end_sec, rng):
        progress = (t - start_sec) / max(0.1, end_sec - start_sec)
        if progress < 0.3:
            return {"t": round(t, 2), "emotion": "neutral", "magnitude": rng.uniform(0.5, 0.8)}
        if progress < 0.6:
            return {"t": round(t, 2), "emotion": rng.choice(["surprise", "neutral", "fear"]),
                    "magnitude": rng.uniform(0.7, 0.9)}
        return {"t": round(t, 2), "emotion": rng.choice(["joy", "anger", "surprise"]),
                "magnitude": rng.uniform(0.8, 1.0)}

    def _collect_emotion_timeseries(self, cap, fps, task, bystanders, climax_sec, window_sec, scored_windows=None):
        """Single decode pass over the union of all bystanders' sampling windows
        (Issue #1). Each bystander's window is anchored to where it is actually
        detected near the climax (Issue #2), and each crop is gated to a real
        face before emotion inference (Issue #3). Returns {person_id: timeseries}
        keeping only series with >= 2 samples."""
        s, e = window_sec
        width = max(0.5, e - s)
        # Issue #2: anchor sampling to bystander presence, not the wearer's climax.
        win = {}
        for b in bystanders:
            pid = b.get('person_id')
            # Cross-layer guardrail — untracked-track filter (docs/03 § Multi-Window
            # Reaction Segments): skip untracked negative-id fragments (03d Resolved
            # #4); they have no real track and, multiplied across segments, inject
            # phantom emotion trajectories.
            if pid is None or pid < 0:
                continue
            ts = b.get('timestamps_sec') or []
            if not ts or not b.get('bounding_boxes'):
                continue
            if any(s - self.MATCH_TOL <= t <= e + self.MATCH_TOL for t in ts):
                w = (s, e)
            else:
                t_near = min(ts, key=lambda t: abs(t - climax_sec))
                # Anchor-distance bound (MAX_ANCHOR_SPAN_SEC): a bystander detected far
                # from this segment's climax was not present at the task — skip it.
                if abs(t_near - climax_sec) > self.MAX_ANCHOR_SPAN_SEC:
                    continue
                w = (t_near, t_near + width)
            # Cross-layer guardrail — distinct-window dedup: a sparse bystander's
            # segments collapse onto the same anchored window; score it once per clip.
            if scored_windows is not None:
                key = (pid, round(w[0], 1), round(w[1], 1))
                if key in scored_windows:
                    continue
                scored_windows.add(key)
            win[pid] = w
        if not win:
            return {}

        # Union of per-bystander sample timestamps -> a single sorted decode pass.
        samples = {}
        for pid, (ws, we) in win.items():
            t = ws
            while t <= we:
                samples.setdefault(round(t, 2), []).append(pid)
                t += self.SAMPLE_DT
        sorted_ts = sorted(samples)
        if not sorted_ts:
            return {}

        bmap = {b.get('person_id'): b for b in bystanders}
        series = {pid: [] for pid in win}
        rng = random.Random(int(sum(sum(bb) for b in bystanders
                                    for bb in (b.get('bounding_boxes') or []))) or 1)

        # Start the sequential decode AT the first sample, not frame 0: on a long
        # clip a late segment would otherwise grab every frame from 0 to the window
        # (~90k frames for a task at t=3000s). One seek + a local grab walk instead.
        first_frame = int(sorted_ts[0] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        if not cap.grab():
            return {}
        cur = first_frame + 1
        for t in sorted_ts:
            target = int(t * fps)
            broke = False
            while cur < target:
                if not cap.grab():
                    broke = True
                    break
                cur += 1
            if broke:
                break
            ok, frame = cap.retrieve()
            cur += 1
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb_faces, meta = [], []
            for pid in samples[t]:
                b = bmap[pid]
                ts, bb = b['timestamps_sec'], b['bounding_boxes']
                i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
                if abs(ts[i] - t) > self.MATCH_TOL:
                    continue
                x1, y1, x2, y2 = bb[i]
                crop = frame[max(0, int(y1) - 20):min(h, int(y2) + 20),
                             max(0, int(x1) - 20):min(w, int(x2) + 20)]
                if crop.size == 0:
                    continue
                if self.emotion_detector is not None:
                    face = self._extract_face(crop)   # Issue #3
                    if face is None:
                        continue
                    rgb_faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    meta.append(pid)
                else:
                    series[pid].append(self._mock_emotion(t, s, e, rng))
            if rgb_faces:
                labels, scores = self._predict_emotions_batch(rgb_faces)   # Issue #1
                for pid, lab, sc in zip(meta, labels, scores):
                    magnitude = float(max(sc))
                    # Confidence gate (Resolved Issue 1): a near-uniform softmax
                    # means HSEmotion can't resolve the face — drop the sample
                    # rather than feed noise downstream.
                    if magnitude < MIN_EMOTION_CONF:
                        continue
                    series[pid].append({
                        "t": round(t, 2),
                        "emotion": HSEMOTION_TO_CANONICAL.get(str(lab).lower(), "neutral"),
                        "magnitude": magnitude,
                    })
        return {pid: ser for pid, ser in series.items() if len(ser) >= 2}
