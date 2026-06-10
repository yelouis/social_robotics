"""Tunable thresholds for the 03c Acoustic Prosody layer.

All seven heuristic constants previously baked into pipeline.py live here as
fields on a frozen dataclass so empirical tuning becomes data, not code.
Override any field via the constructor for tests or A/B runs:

    pipeline = AcousticProsodyPipeline(
        manifest, output,
        config=Layer03cConfig(high_volume_dbfs=-15.0),
    )
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Layer03cConfig:
    # Volume cutoffs for the alarming/discouraging heuristics (dBFS).
    high_volume_dbfs: float = -20.0
    low_volume_dbfs: float = -35.0

    # Additive bonuses applied to the alarming/discouraging scores when the
    # corresponding volume cutoff is crossed.
    high_volume_bonus: float = 0.3
    low_volume_bonus: float = 0.3

    # Multiplier on pitch_contour_variance when computing the soothing score.
    pitch_variance_soothing_weight: float = 0.5

    # Minimum dominant-tone score required for a non-Neutral classification.
    min_dominant_tone_score: float = 0.3

    # SER dominant-emotion confidence below which SenseVoice runs.
    sensevoice_confidence_threshold: float = 0.6

    # Issue 1 (June 9): minimum SER dominant-emotion confidence for ANY
    # non-Neutral tone. Below this the 9-class softmax is effectively guessing
    # (e.g. happy 0.32 vs neutral 0.26 — a coin-flip), so the tone is forced to
    # Neutral instead of letting heuristic bonus terms (pitch variance, volume)
    # fabricate a full-strength classification. Mirrors 03b's MIN_EMOTION_CONF.
    # The same floor gates the pitch-variance term: melody only corroborates a
    # Soothing read when `happy` itself is confident.
    min_ser_confidence: float = 0.5

    # Divisor used to normalize raw pitch variance into [0, 1].
    pitch_variance_normalization: float = 10000.0

    # Issue 2: tasks whose own mechanics are inherently loud (clatter, machinery,
    # power tools). For these, a high-volume reading reflects the *task*, not an
    # alarmed vocal reaction, so the `high_volume_bonus` must NOT push the clip
    # toward "Alarming". Matched as case-insensitive substrings against
    # `task_label`; tuple (not list/set) to keep the dataclass frozen/hashable.
    high_volume_expected_task_keywords: tuple = (
        "cook", "laundry", "clean", "dish", "blacksmith", "construct",
        "renovation", "yardwork", "shovel", "snow", "vacuum", "lawn", "mow",
        "machin", "carpentr", "woodwork", "saw", "drill", "hammer", "weld",
        "grind", "mechanic",
    )
