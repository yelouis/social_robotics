import numpy as np
from collections import deque
import ollama
import shutil
import os

class DistributionMonitor:
    """Task 2: Detects standard deviation collapse in reward signals."""
    def __init__(self, buffer_size=20):
        self.buffer = deque(maxlen=buffer_size)

    def add_score(self, score):
        self.buffer.append(score)

    def check_score_variance_for_collapse(self):
        """If std < 0.1 on a full buffer, trigger ModelCollapseError logic."""
        if len(self.buffer) < self.buffer.maxlen:
            return False
        
        std = np.std(self.buffer)
        if std < 0.1:
            print(f"  [CRITICAL] Standard deviation collapse detected: {std:.4f}")
            return True
        return False

class ConfidenceRouter:
    """Task 3: Routes low-confidence clips for manual review."""
    def __init__(self, review_dir="~/charades_ego_data/manual_review/"):
        self.review_dir = os.path.expanduser(review_dir)
        os.makedirs(self.review_dir, exist_ok=True)

    def check_for_hallucination(self, vlm_state, vlm_conf):
        """
        Uses Gemma 4 to cross-check VLM output quality.
        Returns an adjusted confidence score.
        """
        prompt = (
            f"VLM Output: State={vlm_state}, Confidence={vlm_conf}. "
            "As a safety judge, give a confidence penalty (e.g., -0.2) if this state "
            "seems like a common VLM hallucination for indoor cleaning/cooking. "
            "Reply with ONLY a number."
        )
        try:
            # Using gemma4:latest (the 'Brain' model) to validate the engagement VLM
            response = ollama.chat(
                model='gemma4:latest',
                messages=[{'role': 'user', 'content': prompt}]
            )
            # Basic extraction of the numeric value
            txt = response['message']['content'].strip()
            # Simple float extraction (defensive)
            import re
            match = re.search(r"[-+]?\d*\.\d+|\d+", txt)
            penalty = float(match.group()) if match else 0.0
            return max(0.0, min(1.0, vlm_conf + penalty))
        except Exception as e:
            print(f"  [WARN] Confidence Router error: {e}")
            return vlm_conf

    def route_clip(self, min_confidence, tp_video_path):
        """Flags clip for manual review if confidence falls below threshold."""
        if min_confidence < 0.6:
            clip_name = os.path.basename(tp_video_path)
            dest = os.path.join(self.review_dir, clip_name)
            if not os.path.exists(dest):
                shutil.copy(tp_video_path, dest)
            return True # Flagged
        return False

class ReactionFilter:
    """Task 4: Implements the 'Reaction Lag' temporal filter."""
    @staticmethod
    def validate_reaction_lag(action_start_t, flinch_t):
        """
        Checks if the human flinch happened within a physiologically possible window.
        Returns (is_valid, reason)
        """
        if flinch_t is None or action_start_t is None:
            return False, "Data missing"

        lag_ms = (flinch_t - action_start_t) * 1000
        
        if lag_ms < 100:
            return False, f"Noise: Impossible reaction speed ({lag_ms:.0f}ms)"
        if lag_ms > 5000:
            return False, f"Dropout: Unrelated reaction ({lag_ms:.0f}ms delay)"
        
        return True, f"Valid: {lag_ms:.0f}ms lag"

class ModelCollapseError(Exception):
    """Raised when VLM distribution variance collapses."""
    pass
