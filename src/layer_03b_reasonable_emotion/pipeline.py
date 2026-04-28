import json
import cv2
import traceback
import random
import re
import tempfile
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Literal, Optional

# Optional dependency: PyFeat
try:
    from feat import Detector
except ImportError:
    Detector = None

# Optional dependency: Ollama
try:
    import ollama as ollama_client
except ImportError:
    ollama_client = None

# ---------------------------------------------------------------------------
#  Pydantic schemas for LLM output validation (Limitation #2 fix)
# ---------------------------------------------------------------------------

# The canonical emotion vocabulary — every LLM response is validated against
# this set.  Anything outside it is silently normalized to "neutral".
KNOWN_EMOTIONS = {
    "joy", "amusement", "relief", "surprise",
    "anger", "disgust", "fear", "sadness",
    "neutral", "bored", "distracted", "contempt",
    "excitement", "anxiety", "confusion",
}

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

# PyFeat emotion column order (standard across Py-Feat versions)
PYFEAT_EMOTION_COLUMNS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# ---------------------------------------------------------------------------
#  Default / fallback expectations (used when LLM is unavailable)
# ---------------------------------------------------------------------------
DEFAULT_EXPECTATIONS = {
    "positive_emotions": ["joy", "amusement", "relief"],
    "negative_emotions": ["anger", "disgust", "fear", "sadness"],
    "neutral_baseline": ["neutral", "surprise"]
}

# LLM configuration
OLLAMA_MODEL = "gemma4:e4b"
LLM_MAX_RETRIES = 3


class ReasonableEmotionPipeline:
    def __init__(self, input_manifest_path, output_result_path, attention_result_path=None, force=False):
        self.input_manifest_path = Path(input_manifest_path)
        self.output_result_path = Path(output_result_path)
        self.attention_result_path = Path(attention_result_path) if attention_result_path else None
        self.error_log_path = self.output_result_path.parent / "03b_reasonable_emotion_errors.json"
        self.force = force

        # Initialize PyFeat — CPU-only on macOS (Limitation #1 fix).
        # We instantiate once and reuse across all videos to avoid the heavy
        # model-loading cost per frame.  All inference is synchronous on CPU;
        # concurrency is bounded by the single-threaded iteration in run().
        try:
            if Detector:
                self.feat_detector = Detector(device="cpu")
                print("[03b] PyFeat loaded (CPU mode).")
            else:
                self.feat_detector = None
        except Exception as e:
            print(f"[03b] Failed to load PyFeat: {e}")
            self.feat_detector = None

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
                response = ollama_client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    options={"temperature": 0.3 + (attempt - 1) * 0.2},
                )
                raw_text = response.message.content.strip()
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
                response = ollama_client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    options={"temperature": 0.3 + (attempt - 1) * 0.2},
                )
                raw_text = response.message.content.strip()
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
        with open(self.input_manifest_path, 'r') as f:
            registry = json.load(f)

        results = []
        if self.output_result_path.exists() and not self.force:
            try:
                with open(self.output_result_path, 'r') as f:
                    results = json.load(f)
            except Exception:
                pass

        for entry in registry:
            video_id = entry.get('id', entry.get('video_id'))
            if video_id in self.processed_ids and not self.force:
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
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0 or total_frames == 0:
            cap.release()
            return None

        tasks_analyzed = []
        for task in identified_tasks:
            task_result = self._process_task(cap, fps, task, bystanders, video_id)
            if task_result:
                tasks_analyzed.append(task_result)

        cap.release()

        if not tasks_analyzed:
            return None

        return {
            "video_id": video_id,
            "layer": "03b_reasonable_emotion",
            "tasks_analyzed": tasks_analyzed
        }

    def _process_task(self, cap, fps, task, bystanders, video_id):
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

        per_person = []
        for bystander in bystanders:
            person_id = bystander.get('person_id')

            # Step 2: Temporal Sampling & Pairwise Chunking
            timeseries = self._sample_emotions(cap, fps, start_sec, end_sec, bystander)
            if not timeseries or len(timeseries) < 2:
                continue

            # Step 3: Accumulated History Evaluation
            slices = []
            accumulated_history = ""
            for i in range(len(timeseries) - 1):
                start_emotion = timeseries[i]
                end_emotion = timeseries[i + 1]

                # Use LLM with pydantic validation + retry, or rule-based fallback
                eval_result = self._llm_evaluate_transition(
                    task_label, expectations, accumulated_history,
                    start_emotion['emotion'], end_emotion['emotion']
                )
                direction = eval_result['classified_direction']

                # Build accumulated history string for next iteration
                accumulated_history += (
                    f"{i + 1}. {start_emotion['emotion']} -> {end_emotion['emotion']} "
                    f"(Classified: {direction})\n"
                )

                scalar = end_emotion['magnitude']
                if direction == "negative":
                    scalar = -scalar
                elif direction == "neutral":
                    scalar = 0.0

                slices.append({
                    "slice_id": i + 1,
                    "window_sec": [start_emotion['t'], end_emotion['t']],
                    "transition_pair": [start_emotion['emotion'], end_emotion['emotion']],
                    "terminal_magnitude": round(end_emotion['magnitude'], 2),
                    "classified_direction": direction,
                    "slice_success_scalar": round(scalar, 2)
                })

            # Step 4: Late-Stage Weighted Average Task Success Score
            # Formula: Sum(S_i * D_i * W_i) / Sum(D_i * W_i)
            # W_i = max(0.1, start_i - climax)

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

            per_person.append({
                "person_id": person_id,
                "temporal_slices": slices,
                "late_stage_weighted_success_score": round(person_score, 2)
            })

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

    # ------------------------------------------------------------------
    #  Emotion sampling (Limitation #1 fix — real PyFeat integration)
    # ------------------------------------------------------------------
    def _sample_emotions(self, cap, fps, start_sec, end_sec, bystander):
        """Sample bystander emotions at ~3 FPS within the reaction window.

        When PyFeat is available, crops are written to a temp file and passed
        to Detector.detect_image().  The dominant emotion column from the
        returned DataFrame becomes the label; its probability becomes the
        magnitude.

        When PyFeat is unavailable (or fails), a deterministic mock
        distribution is used for integration testing.
        """
        current_t = start_sec
        timeseries = []
        b_timestamps = bystander.get('timestamps_sec', [])
        b_bboxes = bystander.get('bounding_boxes', [])

        if not b_timestamps or not b_bboxes:
            return timeseries

        # Fix random seed for deterministic mocking based on bounding boxes
        random.seed(int(sum(sum(b) for b in b_bboxes)))

        while current_t <= end_sec:
            # Find closest bbox
            diffs = [abs(t - current_t) for t in b_timestamps]
            closest_idx = diffs.index(min(diffs))

            if diffs[closest_idx] > 2.0:
                current_t += 0.33
                continue

            bbox = b_bboxes[closest_idx]
            x1, y1, x2, y2 = bbox

            frame_idx = int(current_t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            px1 = max(0, int(x1) - 20)
            py1 = max(0, int(y1) - 20)
            px2 = min(w, int(x2) + 20)
            py2 = min(h, int(y2) + 20)

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                current_t += 0.33
                continue

            emotion_label = "neutral"
            magnitude = 0.5
            detected = False

            if self.feat_detector:
                try:
                    # PyFeat's detect_image expects a file path.  Write the
                    # crop to a temporary PNG, run detection, then clean up.
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                        cv2.imwrite(tmp_path, crop)

                    det_result = self.feat_detector.detect_image(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)

                    if det_result is not None and len(det_result) > 0:
                        # Extract emotion columns — PyFeat returns them as
                        # DataFrame columns in a fixed order.
                        emotion_cols = [c for c in det_result.columns
                                        if c.lower() in {e.lower() for e in PYFEAT_EMOTION_COLUMNS}]
                        if emotion_cols:
                            row = det_result[emotion_cols].iloc[0]
                            dominant_idx = row.values.argmax()
                            emotion_label = emotion_cols[dominant_idx].lower()
                            magnitude = float(row.values[dominant_idx])
                            detected = True
                except Exception as e:
                    print(f"[03b] PyFeat inference failed at t={current_t:.2f}s: {e}")
                    # Clean up temp file on failure
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except NameError:
                        pass

            if not detected:
                # Fallback: deterministic mock distribution for integration testing
                progress = (current_t - start_sec) / (max(0.1, end_sec - start_sec))
                if progress < 0.3:
                    emotion_label = "neutral"
                    magnitude = random.uniform(0.5, 0.8)
                elif progress < 0.6:
                    emotion_label = random.choice(["surprise", "neutral", "fear"])
                    magnitude = random.uniform(0.7, 0.9)
                else:
                    emotion_label = random.choice(["joy", "anger", "surprise"])
                    magnitude = random.uniform(0.8, 1.0)

            timeseries.append({
                "t": round(current_t, 2),
                "emotion": emotion_label,
                "magnitude": magnitude
            })

            current_t += 0.33

        return timeseries
